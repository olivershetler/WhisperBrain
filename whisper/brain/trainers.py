import torch
import torch.nn.functional as F
import random
import wandb
import datetime
import pandas as pd
import itertools

from whisper.decoding import DecodingTask
from whisper import DecodingOptions
from whisper.normalizers import EnglishTextNormalizer
import jiwer
import os
import whisper
from pathlib import Path
from whisper.brain.losses import ClipLoss


from abc import abstractmethod, ABC

class BaseTrainer(ABC):
    def __init__(self, model, decoding_options=None, device=None, eval_sample_size=15):
        self.model = model
        if decoding_options is not None:
            if not isinstance(decoding_options, DecodingOptions):
                raise ValueError("decoding_options should be an instance of whisper.DecodingOptions")
            self.decoding_options = decoding_options
        else:
            self.decoding_options = DecodingOptions(
                                        language="en",
                                        without_timestamps=True,
                                        fp16=False,
                                        )
        self.normalizer = EnglishTextNormalizer()
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        model.to(self.device)
        self.timestamp = datetime.datetime.now()
        self.eval_sample_size = eval_sample_size

    def _eval_sample(self, evalloader, step, epoch, sample_size=15):
        self.model.eval()
        with torch.no_grad():
            results = self._eval_predict(evalloader, sample_size)
            brain_wer = jiwer.wer(
                results["reference_clean"],
                results["brain_hypotheses_clean"]
                )
            audio_wer = jiwer.wer(
                results["reference_clean"],
                results["audio_hypotheses_clean"]
                )
            results["step"] = [step]*len(results["sample_nums"])
            results["epoch"] = [epoch]*len(results["sample_nums"])
            # write results to csv in save_dir
            df = pd.DataFrame(results)
            # append to csv
            csv_path = self.save_dir / f"{self.timestamp.strftime('%Y-%m-%d_%H-%M-%S')}_eval_results.csv"
            if os.path.exists(csv_path):
                df.to_csv(csv_path, mode="a", header=False)
            else:
                df.to_csv(csv_path, header=True)
            dc_results = wandb.Table(dataframe=df)
            wandb.log({"eval": {
                "brain_wer": brain_wer,
                "audio_wer": audio_wer,
                "step": step,
                "epoch": epoch,
                "dc_results": dc_results
                }})
        self.model.train()

    def _eval_predict(self, evalloader, sample_size=15):
        brain_hypotheses = []
        audio_hypotheses = []
        references = []
        sample_nums = []
        i = 0
        for brain_data, mels, texts, _ in evalloader:

            self.model.toggle_mode("brain")
            neural_results = self._decode(brain_data)
            brain_hypotheses.extend([result.text for result in neural_results])

            self.model.toggle_mode("audio")
            audio_results = self._decode(mels)
            audio_hypotheses.extend([result.text for result in audio_results])
            references.extend(texts)
            sample_nums.extend([i]*len(texts))

            if i == sample_size:
                break
            i += 1

        data = {
            "sample_nums": sample_nums,
            "brain_hypotheses": brain_hypotheses,
            "audio_hypotheses": audio_hypotheses,
            "reference": references,
            "brain_hypotheses_clean": [self.normalizer(text) for text in brain_hypotheses],
            "audio_hypotheses_clean": [self.normalizer(text) for text in audio_hypotheses],
            "reference_clean": [self.normalizer(text) for text in references],
        }
        return data

    @torch.no_grad()
    def _decode(self, brain_data):
        result = DecodingTask(self.model, self.decoding_options).run(brain_data)
        return result

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def save_checkpoint(self, *args, **kwargs):
        pass


class BigAssTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        model_optimizer,
        model_metaoptimizer,
        embedding_critic,
        embedding_critic_optimizer,
        logit_critic,
        logit_critic_optimizer,
        checkpoint_dir,
        default_forcing_prob=1,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        model_scheduler=None,
        embedding_critic_scheduler=None,
        logit_critic_scheduler=None,
        critic_scheduler=None,
        max_grad_norm=1,
        normalizer=None,
        tokenizer=None,
        decoding_options=None,
        device=None,
        eval_sample_size=15
        ):
        super().__init__(model, decoding_options, device=device, eval_sample_size=eval_sample_size)

        self.trainer_name = "big-ass-trainer"

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler
        self.embedding_critic = embedding_critic
        self.embedding_critic_optimizer = embedding_critic_optimizer
        self.embedding_critic_scheduler = embedding_critic_scheduler
        self.logit_critic = logit_critic
        self.logit_critic_optimizer = logit_critic_optimizer
        self.logit_critic_scheduler = logit_critic_scheduler

        self.critic_scheduler = critic_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        self.forcing_prob = default_forcing_prob

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")

        self._bce_loss = torch.nn.BCELoss(reduction="mean").to(self.device)
        self._ce_loss = torch.nn.CrossEntropyLoss().to(self.device)

        self.start_time = datetime.datetime.now()

        run_dir = f"{self.trainer_name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(project="whisperbrain", name=f"{run_dir}")
        self.save_dir = self.checkpoint_dir / run_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_model_checkpoint(self, epoch, step):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.model_optimizer.state_dict(),
            "scheduler": self.model_scheduler.state_dict() if self.model_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        pth = self.save_dir / f"{run_ts}_{epoch}_{step}_whisperbrain_checkpoint.pt"
        torch.save(state_dict, pth)

    def _save_embedding_critic_checkpoint(self, epoch, step):
        state_dict = {
            "critic": self.embedding_critic.state_dict(),
            "optimizer": self.embedding_critic_optimizer.state_dict(),
            "scheduler": self.embedding_critic_scheduler.state_dict() if self.embedding_critic_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(state_dict, self.save_dir / f"{run_ts}_{epoch}_{step}_embedding_critic_checkpoint.pt")

    def _save_logit_critic_checkpoint(self, epoch, step):
        state_dict = {
            "critic": self.logit_critic.state_dict(),
            "optimizer": self.logit_critic_optimizer.state_dict(),
            "scheduler": self.logit_critic_scheduler.state_dict() if self.logit_critic_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(state_dict, self.save_dir / f"{run_ts}_{epoch}_{step}_logit_critic_checkpoint.pt")

    def save_checkpoint(self, epoch, step):
        self._save_model_checkpoint(epoch, step)
        self._save_embedding_critic_checkpoint(epoch, step)
        self._save_logit_critic_checkpoint(epoch, step)

    def load_checkpoints(self, model_path, embedding_critic_path, logit_critic_path):
        model_state_dict = torch.load(model_path)
        self.model.load_state_dict(model_state_dict["model"])
        self.model_optimizer.load_state_dict(model_state_dict["optimizer"])
        if self.model_scheduler is not None:
            self.model_scheduler.load_state_dict(model_state_dict["scheduler"])

        embedding_critic_state_dict = torch.load(embedding_critic_path)
        self.embedding_critic.load_state_dict(embedding_critic_state_dict["critic"])
        self.embedding_critic_optimizer.load_state_dict(embedding_critic_state_dict["optimizer"])
        if self.embedding_critic_scheduler is not None:
            self.embedding_critic_scheduler.load_state_dict(embedding_critic_state_dict["scheduler"])

        logit_critic_state_dict = torch.load(logit_critic_path)
        self.logit_critic.load_state_dict(logit_critic_state_dict["critic"])
        self.logit_critic_optimizer.load_state_dict(logit_critic_state_dict["optimizer"])
        if self.logit_critic_scheduler is not None:
            self.logit_critic_scheduler.load_state_dict(logit_critic_state_dict["scheduler"])

    def _encode_texts(self, texts):
        cleaned_texts = map(self.normalizer, texts)
        tokens_list = []
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        eot = [self.tokenizer.eot]
        for text in cleaned_texts:
            tokens = sots + list(self.tokenizer.encode(text)) + eot
            tokens_list.append(tokens)
        if len(tokens_list) > 1:
            padtoken = 50256
            max_len = max([len(tokens) for tokens in tokens_list])
            for i, tokens in enumerate(tokens_list):
                if len(tokens) < max_len:
                    tokens_list[i] = tokens + [padtoken] * (max_len - len(tokens))
        return torch.tensor(tokens_list, device=self.device)

    def _supplemental_embedding_loss(self, brain_embeddings, audio_embeddings):
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        N, S, D = brain_embeddings.shape
        x = brain_embeddings.view(N, S*D)
        y = audio_embeddings.view(N, S*D)
        return loss_fn(x, y, torch.ones(N, device=x.device))

    def _supplemental_logit_loss(self, brain_logits, audio_logits):
        # kl divergence loss
        N, S, V = brain_logits.shape
        x = F.softmax(brain_logits.view(N*S, V), dim=-1)
        y = F.log_softmax(audio_logits.view(N*S, V), dim=-1)
        kldiv = torch.nn.KLDivLoss(reduction="batchmean")
        return kldiv(y, x)

    def _embedding_gan_generator_loss(self, brain_embeddings):
        # Pass brain embeddings through critic
        fake_logits = self.embedding_critic(brain_embeddings)
        # Generator loss (flip labels for the generator)
        embedding_g_loss = self._bce_loss(fake_logits, torch.ones_like(fake_logits, device=fake_logits.device))
        return embedding_g_loss

    def _embedding_gan_critic_loss(self, brain_embeddings, audio_embeddings):
        # Create mixed embeddings and ground truth labels
        N, S, D = brain_embeddings.shape
        new_shape = (N * 2, S, D)
        mixed_embeddings = torch.zeros(new_shape).to(self.device)
        ground_truth = torch.zeros((N * 2, 1)).to(self.device)
        index = [(j, 'audio') for j in range(N)] + [(j, 'brain') for j in range(N)]
        random.shuffle(index)
        for i, (j, source) in enumerate(index):
            if source == 'audio':
                mixed_embeddings[i] = audio_embeddings[j]
                ground_truth[i] = 1
            else:
                mixed_embeddings[i] = brain_embeddings[j]
                ground_truth[i] = 0
        # critic forward pass
        logits = self.embedding_critic(mixed_embeddings).to(self.device)
        bce = torch.nn.BCELoss(reduction="mean").to(self.device)
        embedding_d_loss = bce(logits, ground_truth)
        return embedding_d_loss

    def _logit_gan_partial_generator_loss(self, brain_logits):
        fake_critic_logits = self.logit_critic(brain_logits)
        g_logit_loss = self._bce_loss(fake_critic_logits, torch.ones_like(fake_critic_logits))
        return g_logit_loss

    def _logit_gan_partial_critic_loss(self, brain_logits, audio_logits):
        # Create mixed logits batch
        N, S, V = brain_logits.shape
        new_shape = (N * 2, S, V)
        mixed_logits = torch.zeros(new_shape).to(self.device)
        ground_truth = torch.zeros((N * 2, 1)).to(self.device)
        index = [(j, 'audio') for j in range(N)] + [(j, 'brain') for j in range(N)]
        random.shuffle(index)
        for i, (j, source) in enumerate(index):
            if source == 'audio':
                mixed_logits[i] = audio_logits[j]
                ground_truth[i] = 1
            else:
                mixed_logits[i] = brain_logits[j]
                ground_truth[i] = 0
        # critic forward pass
        real_critic_logits = self.logit_critic(mixed_logits)
        d_logit_loss = self._bce_loss(real_critic_logits, ground_truth)
        return d_logit_loss

    def _sequence_partial_loss(self, brain_logits, tokens, t):
        ce_losses = []
        for i in range(tokens.size(0)):
            x = brain_logits[i, :, :]
            y_index = tokens[i, :t+1]
            ce_losses += [self._ce_loss(x, y_index)]
        return torch.stack(ce_losses).mean()

    def _propogate_model_losses(self, embedding_g_loss, sup_embedding_loss, logit_g_loss, sup_logit_loss, seq_ce_loss, retain_graph=False):
        # first propogate generator
        embedding_g_loss.backward(retain_graph=True)
        sup_embedding_loss.backward(retain_graph=True)
        logit_g_loss.backward(retain_graph=True)
        sup_logit_loss.backward(retain_graph=True)
        seq_ce_loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()
        self.model_optimizer.zero_grad()

    def _propogate_embedding_critic_loss(self, embedding_d_loss, retain_graph=False):
        embedding_d_loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.embedding_critic.parameters(), self.max_grad_norm)
        self.embedding_critic_optimizer.step()
        self.embedding_critic_optimizer.zero_grad()

    def _propogate_logit_critic_loss(self, logit_d_loss, retain_graph=False):
        logit_d_loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.logit_critic.parameters(), self.max_grad_norm)
        self.logit_critic_optimizer.step()
        self.logit_critic_optimizer.zero_grad()

    def _model_train_step(self, batch, epoch, logging=True):
        self.model_optimizer.zero_grad()
        self.embedding_critic_optimizer.zero_grad()
        self.logit_critic_optimizer.zero_grad()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)

        embedding_g_loss = self._embedding_gan_generator_loss(brain_embeddings)
        sup_embedding_loss = self._supplemental_embedding_loss(brain_embeddings, audio_embeddings)

        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        logit_g_losses = []
        seq_ce_losses = []
        sup_logit_losses = []

        brain_generated_tokens = sots.copy()
        audio_generated_tokens = sots.copy()
        for t in range(start_t, tokens.size(1)):
            if random.random() >= self.forcing_prob and start_t > len(sots):
                brain_input = torch.tensor([brain_generated_tokens], device=brain_embeddings.device)
                audio_input = torch.tensor([audio_generated_tokens], device=audio_embeddings.device)
            else:
                brain_input = tokens[:, :t+1]
                audio_input = tokens[:, :t+1]
            brain_logits = self.model.logits(brain_input, brain_embeddings)
            audio_logits = self.model.logits(audio_input, audio_embeddings)
            brain_generated_tokens += [torch.argmax(brain_logits[:, -1, :]).item()]
            audio_generated_tokens += [torch.argmax(audio_logits[:, -1, :]).item()]
            logit_g_losses += [self._logit_gan_partial_generator_loss(brain_logits)]
            sup_logit_losses += [self._supplemental_logit_loss(brain_logits, audio_logits)]
            seq_ce_losses += [self._sequence_partial_loss(brain_logits, tokens, t)]

        logit_g_loss = torch.stack(logit_g_losses).mean()
        seq_ce_loss = torch.stack(seq_ce_losses).mean()
        sup_logit_loss = torch.stack(sup_logit_losses).mean()

        self._propogate_model_losses(
            embedding_g_loss,
            sup_embedding_loss,
            logit_g_loss,
            sup_logit_loss,
            seq_ce_loss,
            retain_graph=False
            )

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()

        if logging:
            wandb.log({"train": {
                "epoch": epoch,
                "learning_rate": self.model_optimizer.param_groups[0]["lr"],
                "embedding_generator_loss": embedding_g_loss,
                "embedding_cos_loss": sup_embedding_loss,
                "logit_generator_loss": logit_g_loss,
                "logit_kldiv_loss": sup_logit_loss,
                "sequence_crossentropy": seq_ce_loss
                }})


    def _train_step(self, batch, epoch, logging=True):
        self.model.train()
        self._model_train_step(batch, epoch, logging=logging)
        self.model.eval()
        self._embedding_critic_train_step(batch, epoch, logging=logging)
        self.model.eval()
        self._logit_critic_train_step(batch, epoch, logging=logging)

    def _step_epoch(self, k, batch, evalloader, epoch):
        self._train_step(batch, epoch, logging=k % self.logging_interval == 0)
        if k % self.checkpoint_interval == 0:
            self.save_checkpoint(epoch, k)
        if k % self.eval_interval == 0 and evalloader is not None:
            self._eval_sample(evalloader, k, epoch, sample_size=self.eval_sample_size)

    def train(self, trainloader, evalloader=None, epoch=1, forcing_prob=1):
        self.model.train()
        self.forcing_prob = forcing_prob
        for k, batch in enumerate(trainloader):
            self._step_epoch(k, batch, evalloader, epoch)
        self.model.train()

class FastBigAssTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        model_optimizer,
        embedding_critic,
        embedding_critic_optimizer,
        logit_critic,
        logit_critic_optimizer,
        checkpoint_dir,
        model_loss_weights={
            "embedding_g_loss": 1,
            "sup_embedding_loss": 1,
            "logit_g_loss": 1,
            "sup_logit_loss": 1,
            "seq_ce_loss": 0.1
            },
        default_forcing_prob=1,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        model_scheduler=None,
        embedding_critic_scheduler=None,
        logit_critic_scheduler=None,
        critic_scheduler=None,
        max_grad_norm=1,
        normalizer=None,
        tokenizer=None,
        decoding_options=None,
        device=None,
        eval_sample_size=15
        ):
        super().__init__(model, decoding_options, device=device, eval_sample_size=eval_sample_size)

        self.trainer_name = "big-ass-trainer"

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler
        self.model_loss_weights = model_loss_weights
        self.embedding_critic = embedding_critic
        self.embedding_critic_optimizer = embedding_critic_optimizer
        self.embedding_critic_scheduler = embedding_critic_scheduler
        self.logit_critic = logit_critic
        self.logit_critic_optimizer = logit_critic_optimizer
        self.logit_critic_scheduler = logit_critic_scheduler

        self.critic_scheduler = critic_scheduler
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        self.forcing_prob = default_forcing_prob

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")

        self._bce_loss = torch.nn.BCELoss(reduction="mean").to(self.device)
        self._ce_loss = torch.nn.CrossEntropyLoss().to(self.device)

        self.start_time = datetime.datetime.now()

        run_dir = f"{self.trainer_name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(project="whisperbrain", name=f"{run_dir}")
        self.save_dir = self.checkpoint_dir / run_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _save_model_checkpoint(self, epoch, step):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.model_optimizer.state_dict(),
            "scheduler": self.model_scheduler.state_dict() if self.model_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        pth = self.save_dir / f"{run_ts}_{epoch}_{step}_whisperbrain_checkpoint.pt"
        torch.save(state_dict, pth)

    def _save_embedding_critic_checkpoint(self, epoch, step):
        state_dict = {
            "critic": self.embedding_critic.state_dict(),
            "optimizer": self.embedding_critic_optimizer.state_dict(),
            "scheduler": self.embedding_critic_scheduler.state_dict() if self.embedding_critic_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(state_dict, self.save_dir / f"{run_ts}_{epoch}_{step}_embedding_critic_checkpoint.pt")

    def _save_logit_critic_checkpoint(self, epoch, step):
        state_dict = {
            "critic": self.logit_critic.state_dict(),
            "optimizer": self.logit_critic_optimizer.state_dict(),
            "scheduler": self.logit_critic_scheduler.state_dict() if self.logit_critic_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        torch.save(state_dict, self.save_dir / f"{run_ts}_{epoch}_{step}_logit_critic_checkpoint.pt")

    def save_checkpoint(self, epoch, step):
        self._save_model_checkpoint(epoch, step)
        self._save_embedding_critic_checkpoint(epoch, step)
        self._save_logit_critic_checkpoint(epoch, step)

    def load_checkpoints(self, model_path, embedding_critic_path, logit_critic_path):
        model_state_dict = torch.load(model_path)
        self.model.load_state_dict(model_state_dict["model"])
        self.model_optimizer.load_state_dict(model_state_dict["optimizer"])
        if self.model_scheduler is not None:
            self.model_scheduler.load_state_dict(model_state_dict["scheduler"])

        embedding_critic_state_dict = torch.load(embedding_critic_path)
        self.embedding_critic.load_state_dict(embedding_critic_state_dict["critic"])
        self.embedding_critic_optimizer.load_state_dict(embedding_critic_state_dict["optimizer"])
        if self.embedding_critic_scheduler is not None:
            self.embedding_critic_scheduler.load_state_dict(embedding_critic_state_dict["scheduler"])

        logit_critic_state_dict = torch.load(logit_critic_path)
        self.logit_critic.load_state_dict(logit_critic_state_dict["critic"])
        self.logit_critic_optimizer.load_state_dict(logit_critic_state_dict["optimizer"])
        if self.logit_critic_scheduler is not None:
            self.logit_critic_scheduler.load_state_dict(logit_critic_state_dict["scheduler"])

    def _encode_texts(self, texts):
        cleaned_texts = map(self.normalizer, texts)
        tokens_list = []
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        eot = [self.tokenizer.eot]
        for text in cleaned_texts:
            tokens = sots + list(self.tokenizer.encode(text)) + eot
            tokens_list.append(tokens)
        if len(tokens_list) > 1:
            padtoken = 50256
            max_len = max([len(tokens) for tokens in tokens_list])
            for i, tokens in enumerate(tokens_list):
                if len(tokens) < max_len:
                    tokens_list[i] = tokens + [padtoken] * (max_len - len(tokens))
        return torch.tensor(tokens_list, device=self.device)

    def _supplemental_embedding_loss(self, brain_embeddings, audio_embeddings):
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        N, S, D = brain_embeddings.shape
        x = brain_embeddings.view(N, S*D)
        y = audio_embeddings.view(N, S*D)
        return loss_fn(x, y, torch.ones(N, device=x.device))

    def _supplemental_logit_loss(self, brain_logits, audio_logits):
        # kl divergence loss
        N, S, V = brain_logits.shape
        x = F.softmax(brain_logits.view(N*S, V), dim=-1)
        y = F.log_softmax(audio_logits.view(N*S, V), dim=-1)
        kldiv = torch.nn.KLDivLoss(reduction="batchmean")
        return kldiv(y, x)

    def _embedding_gan_generator_loss(self, brain_embeddings):
        # Pass brain embeddings through critic
        fake_logits = self.embedding_critic(brain_embeddings)
        # Generator loss (flip labels for the generator)
        embedding_g_loss = self._bce_loss(fake_logits, torch.ones_like(fake_logits, device=fake_logits.device))
        return embedding_g_loss

    def _embedding_gan_critic_loss(self, brain_embeddings, audio_embeddings):
        # Create mixed embeddings and ground truth labels
        N, S, D = brain_embeddings.shape
        new_shape = (N * 2, S, D)
        mixed_embeddings = torch.zeros(new_shape).to(self.device)
        ground_truth = torch.zeros((N * 2, 1)).to(self.device)
        index = [(j, 'audio') for j in range(N)] + [(j, 'brain') for j in range(N)]
        random.shuffle(index)
        for i, (j, source) in enumerate(index):
            if source == 'audio':
                mixed_embeddings[i] = audio_embeddings[j]
                ground_truth[i] = 1
            else:
                mixed_embeddings[i] = brain_embeddings[j]
                ground_truth[i] = 0
        # critic forward pass
        logits = self.embedding_critic(mixed_embeddings).to(self.device)
        bce = torch.nn.BCELoss(reduction="mean").to(self.device)
        embedding_d_loss = bce(logits, ground_truth)
        return embedding_d_loss

    def _logit_gan_partial_generator_loss(self, brain_logits):
        fake_critic_logits = self.logit_critic(brain_logits)
        g_logit_loss = self._bce_loss(fake_critic_logits, torch.ones_like(fake_critic_logits))
        return g_logit_loss

    def _logit_gan_partial_critic_loss(self, brain_logits, audio_logits):
        # Create mixed logits batch
        N, S, V = brain_logits.shape
        new_shape = (N * 2, S, V)
        mixed_logits = torch.zeros(new_shape).to(self.device)
        ground_truth = torch.zeros((N * 2, 1)).to(self.device)
        index = [(j, 'audio') for j in range(N)] + [(j, 'brain') for j in range(N)]
        random.shuffle(index)
        for i, (j, source) in enumerate(index):
            if source == 'audio':
                mixed_logits[i] = audio_logits[j]
                ground_truth[i] = 1
            else:
                mixed_logits[i] = brain_logits[j]
                ground_truth[i] = 0
        # critic forward pass
        real_critic_logits = self.logit_critic(mixed_logits)
        d_logit_loss = self._bce_loss(real_critic_logits, ground_truth)
        return d_logit_loss


    def _propogate_model_losses(self, embedding_g_loss, sup_embedding_loss, logit_g_loss, sup_logit_loss, seq_ce_loss, retain_graph=False):
        losses = {
            "embedding_g_loss": embedding_g_loss,
            "sup_embedding_loss": sup_embedding_loss,
            "logit_g_loss": logit_g_loss,
            "sup_logit_loss": sup_logit_loss,
            "seq_ce_loss": seq_ce_loss
        }
        loss = sum([losses[k] * self.model_loss_weights[k] for k in losses.keys()])
        loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()
        self.model_optimizer.zero_grad()

    def _propogate_embedding_critic_loss(self, embedding_d_loss, retain_graph=False):
        embedding_d_loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.embedding_critic.parameters(), self.max_grad_norm)
        self.embedding_critic_optimizer.step()
        self.embedding_critic_optimizer.zero_grad()

    def _propogate_logit_critic_loss(self, logit_d_loss, retain_graph=False):
        logit_d_loss.backward(retain_graph=retain_graph)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.logit_critic.parameters(), self.max_grad_norm)
        self.logit_critic_optimizer.step()
        self.logit_critic_optimizer.zero_grad()

    def _model_train_step(self, batch, epoch, logging=True):
        self.model_optimizer.zero_grad()
        self.embedding_critic_optimizer.zero_grad()
        self.logit_critic_optimizer.zero_grad()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)

        embedding_g_loss = self._embedding_gan_generator_loss(brain_embeddings)
        embedding_d_loss = self._embedding_gan_critic_loss(brain_embeddings.detach(), audio_embeddings.detach())
        sup_embedding_loss = self._supplemental_embedding_loss(brain_embeddings, audio_embeddings)

        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        logit_d_losses = []
        logit_g_losses = []
        seq_ce_losses = []
        sup_logit_losses = []

        brain_generated_tokens = sots.copy()
        audio_generated_tokens = sots.copy()

        for t in range(start_t, tokens.size(1)):
            if random.random() >= self.forcing_prob and start_t > len(sots):
                brain_input = torch.tensor([brain_generated_tokens], device=brain_embeddings.device)
                audio_input = torch.tensor([audio_generated_tokens], device=audio_embeddings.device)
            else:
                brain_input = tokens[:, :t+1]
                audio_input = tokens[:, :t+1]
            brain_logits = self.model.logits(brain_input, brain_embeddings)
            audio_logits = self.model.logits(audio_input, audio_embeddings)
            brain_generated_tokens.append([torch.argmax(brain_logits[:, -1, :]).item()])
            audio_generated_tokens.append([torch.argmax(audio_logits[:, -1, :]).item()])

            logit_g_losses.append(self._logit_gan_partial_generator_loss(brain_logits))
            logit_d_losses.append(self._logit_gan_partial_critic_loss(brain_logits.detach(), audio_logits.detach()))
            sup_logit_losses.append(self._supplemental_logit_loss(brain_logits, audio_logits))
            num_classes = brain_logits.size(-1)
            assert num_classes == 51864
            ce_targets = F.one_hot(tokens[:, t+1], num_classes=num_classes).to(self.device)
            seq_ce_losses.append(self._ce_loss(brain_logits, ce_targets))

        logit_g_loss = torch.stack(logit_g_losses).mean()
        logit_d_loss = torch.stack(logit_d_losses).mean()
        seq_ce_loss = torch.stack(seq_ce_losses).mean()
        sup_logit_loss = torch.stack(sup_logit_losses).mean()

        self._propogate_model_losses(
            embedding_g_loss=embedding_g_loss,
            sup_embedding_loss=sup_embedding_loss,
            logit_g_loss=logit_g_loss,
            sup_logit_loss=sup_logit_loss,
            seq_ce_loss=seq_ce_loss,
            retain_graph=False
            )
        self._propogate_embedding_critic_loss(embedding_d_loss, retain_graph=False)
        self._propogate_logit_critic_loss(logit_d_loss, retain_graph=False)

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()
        self.embedding_critic_optimizer.step()
        self.logit_critic_optimizer.step()

        if logging:
            wandb.log({"train": {
                "epoch": epoch,
                "learning_rate": self.model_optimizer.param_groups[0]["lr"],
                "embedding_generator_loss": embedding_g_loss,
                "embedding_critic_loss": embedding_d_loss,
                "embedding_cos_loss": sup_embedding_loss,
                "logit_generator_loss": logit_g_loss,
                "logit_critic_loss": logit_d_loss,
                "logit_kldiv_loss": sup_logit_loss,
                "sequence_crossentropy": seq_ce_loss
                }})

    def _logit_critic_train_step(self, batch, epoch, logging=True):
        self.model_optimizer.zero_grad()
        self.embedding_critic_optimizer.zero_grad()
        self.logit_critic_optimizer.zero_grad()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)

        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        logit_d_losses = []

        brain_generated_tokens = sots.copy()
        audio_generated_tokens = sots.copy()
        for t in range(start_t, tokens.size(1)):
            if random.random() >= self.forcing_prob and start_t > len(sots):
                brain_input = torch.tensor([brain_generated_tokens], device=brain_embeddings.device)
                audio_input = torch.tensor([audio_generated_tokens], device=audio_embeddings.device)
            else:
                brain_input = tokens[:, :t+1]
                audio_input = tokens[:, :t+1]
            brain_logits = self.model.logits(brain_input, brain_embeddings)
            audio_logits = self.model.logits(audio_input, audio_embeddings)
            brain_generated_tokens += [torch.argmax(brain_logits[:, -1, :]).item()]
            audio_generated_tokens += [torch.argmax(audio_logits[:, -1, :]).item()]

            logit_d_losses += [self._logit_gan_partial_critic_loss(brain_logits, audio_logits)]

        logit_d_loss = torch.stack(logit_d_losses).mean()

        self._propogate_logit_critic_loss(logit_d_loss, retain_graph=False)

        if logging:
            wandb.log({"train": {
                "epoch": epoch,
                "learning_rate": self.model_optimizer.param_groups[0]["lr"],
                "logit_critic_loss": logit_d_loss,
                }})

    def _embedding_critic_train_step(self, batch, epoch, logging=False):
        self.model_optimizer.zero_grad()
        self.embedding_critic_optimizer.zero_grad()
        self.logit_critic_optimizer.zero_grad()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)

        embedding_d_loss = self._embedding_gan_critic_loss(brain_embeddings, audio_embeddings)

        self._propogate_embedding_critic_loss(embedding_d_loss, retain_graph=True)

        if logging:
            wandb.log({"train": {
                "epoch": epoch,
                "learning_rate": self.model_optimizer.param_groups[0]["lr"],
                "embedding_critic_loss": embedding_d_loss,
                }})


    def _step_epoch(self, k, batch, evalloader, epoch):
        self._train_step(batch, epoch, logging=k % self.logging_interval == 0)
        if k % self.checkpoint_interval == 0:
            self.save_checkpoint(epoch, k)
        if k % self.eval_interval == 0 and evalloader is not None:
            self._eval_sample(evalloader, k, epoch, sample_size=self.eval_sample_size)

    def train(self, trainloader, evalloader=None, epoch=1, forcing_prob=1):
        self.model.train()
        self.forcing_prob = forcing_prob
        for k, batch in enumerate(trainloader):
            self._step_epoch(k, batch, evalloader, epoch)
        self.model.train()

class TrainerInterweaver:
    def __init__(self, trainers):
        self.trainers = trainers

    def train(self, trainloader, evalloader=None, epochs=1):
        for i in range(epochs):
            for k, batch in enumerate(trainloader):
                for trainer in self.trainers:
                    trainer._step_epoch(k, batch, trainloader, evalloader, i)



class CLIPTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        model_optimizer,
        checkpoint_dir,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        forcing_prob=1,
        model_scheduler=None,
        max_grad_norm=1,
        normalizer=None,
        tokenizer=None,
        decoding_options=None,
        device=None,
        eval_sample_size=15
        ):
        super().__init__(model, decoding_options, device=device, eval_sample_size=eval_sample_size)

        self.trainer_name = "clip-trainer"

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_scheduler = model_scheduler

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")
        self.forcing_prob = forcing_prob

        self._ce_loss = torch.nn.CrossEntropyLoss().to(self.device)
        self._mse_loss = torch.nn.MSELoss().to(self.device)

        self.start_time = datetime.datetime.now()

        run_dir = f"{self.trainer_name}_{self.start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
        wandb.init(project="whisperbrain", name=f"{self.trainer_name}--{self.start_time.strftime('%Y-%m-%d--%H-%M-%S')}")
        self.save_dir = self.checkpoint_dir / run_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self._embedding_clip_loss = ClipLoss()
        self._logit_clip_loss = ClipLoss()

    def _save_model_checkpoint(self, epoch, step):
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.model_optimizer.state_dict(),
            "scheduler": self.model_scheduler.state_dict() if self.model_scheduler is not None else None
        }
        run_ts = self.start_time.strftime("%Y-%m-%d_%H-%M-%S")
        pth = self.save_dir / f"{run_ts}_{epoch}_{step}_whisperbrain_checkpoint.pt"
        torch.save(state_dict, pth)

    def save_checkpoint(self, epoch, step):
        self._save_model_checkpoint(epoch, step)

    def load_checkpoint(self, model_path):
        model_state_dict = torch.load(model_path)
        self.model.load_state_dict(model_state_dict["model"])
        self.model_optimizer.load_state_dict(model_state_dict["optimizer"])
        if self.model_scheduler is not None:
            self.model_scheduler.load_state_dict(model_state_dict["scheduler"])

    def _encode_texts(self, texts):
        cleaned_texts = map(self.normalizer, texts)
        tokens_list = []
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        eot = [self.tokenizer.eot]
        for text in cleaned_texts:
            tokens = sots + list(self.tokenizer.encode(text)) + eot
            tokens_list.append(tokens)
        if len(tokens_list) > 1:
            padtoken = 50256
            max_len = max([len(tokens) for tokens in tokens_list])
            for i, tokens in enumerate(tokens_list):
                if len(tokens) < max_len:
                    tokens_list[i] = tokens + [padtoken] * (max_len - len(tokens))
        return torch.tensor(tokens_list, device=self.device)

    def _train_step(self, batch, epoch, logging=True):
        self.model.train()
        self.model_optimizer.zero_grad()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings= self.model.embed_brain(brain_data)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)

        embedding_clip_loss = self._embedding_clip_loss(brain_embeddings.permute(0,2,1), audio_embeddings.permute(0,2,1))
        embedding_mse_loss = self._mse_loss(brain_embeddings, audio_embeddings)
        tokens = self._encode_texts(texts).to(self.device)

        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        logit_clip_loss = 0
        logit_mse_loss = 0
        logit_ce_loss = 0

        brain_generated_tokens = sots.copy()
        audio_generated_tokens = sots.copy()
        for t in range(start_t, tokens.size(1)):
            print(f"t={t+1}/{tokens.size(1)}", end="\r")
            if random.random() >= self.forcing_prob and start_t > len(sots):
                brain_input = torch.tensor([brain_generated_tokens], device=brain_embeddings.device)
                audio_input = torch.tensor([audio_generated_tokens], device=audio_embeddings.device)
            else:
                brain_input = tokens[:, :t]
                audio_input = tokens[:, :t]
            brain_logits = self.model.logits(brain_input, brain_embeddings)
            audio_logits = self.model.logits(audio_input, audio_embeddings)
            brain_generated_tokens += [torch.argmax(brain_logits[:, -1, :]).item()]
            audio_generated_tokens += [torch.argmax(audio_logits[:, -1, :]).item()]
            logit_clip_loss += self._logit_clip_loss(brain_logits.permute(0,2,1), audio_logits.permute(0,2,1))
            logit_mse_loss += self._mse_loss(brain_logits.permute(0,2,1), audio_logits.permute(0,2,1))
            token_indices = tokens[:, t]
            num_classes = brain_logits.size(-1)
            assert num_classes == 51864
            logit_ce_loss += self._ce_loss(brain_logits.permute(0,2,1), audio_logits.permute(0,2,1))

        logit_clip_loss /= tokens.size(1)
        logit_mse_loss /= tokens.size(1)
        logit_ce_loss /= tokens.size(1)

        loss = embedding_clip_loss + embedding_mse_loss + logit_clip_loss + logit_mse_loss + logit_ce_loss
        loss.backward(retain_graph=False)
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.model_optimizer.step()

        if logging:
            wandb.log({"train": {
                "epoch": epoch,
                "learning_rate": self.model_optimizer.param_groups[0]["lr"],
                "forcing_prob": self.forcing_prob,
                "embedding_clip_loss": embedding_clip_loss,
                "embedding_mse_loss": embedding_mse_loss,
                "logit_clip_loss": logit_clip_loss,
                "logit_mse_loss": logit_mse_loss,
                "logit_ce_loss": logit_ce_loss
                }})

    def _step_epoch(self, k, batch, evalloader, epoch):
        self._train_step(batch, epoch, logging=k % self.logging_interval == 0)
        if k % self.checkpoint_interval == 0:
            self.save_checkpoint(epoch, k)
        if k % self.eval_interval == 0 and evalloader is not None:
            self._eval_sample(evalloader, k, epoch, sample_size=self.eval_sample_size)

    def train(self, trainloader, evalloader=None, epoch=1, forcing_probs=[1, 0.95]):
        for k, batch in enumerate(trainloader):
            self.forcing_prob = random.uniform(*forcing_probs)
            self._step_epoch(k, batch, evalloader, epoch)