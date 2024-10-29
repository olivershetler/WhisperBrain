class Trainer(BaseTrainer):
    def __init__(
        self,
        trainer_name,
        model,
        optimizer,
        checkpoint_dir,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1):

        super().__init__(model, decoding_options, device=device, eval_sample_size=eval_sample_size)

        self.trainer_name = trainer_name

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps=gradient_accumulation_steps


    def train(self, trainloader, evalloader=None, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            self.train_epoch(trainloader, evalloader, epoch)


    def train_epoch(self, trainloader, evalloader=None, epoch=0, *args, **kwargs):
        for i, batch in enumerate(trainloader):
            self._step_epoch(i, batch, trainloader, evalloader, epoch, *args, **kwargs)

    def _step_epoch(self, i, batch, trainloader, evalloader, epoch, *args, **kwargs):
        if i % self.logging_interval == 0 or i+1 == len(trainloader):
            logging = True
        else:
            logging = False
        loss = self._train_step(batch, logging, epoch, i, *args, **kwargs)
        if i % self.gradient_accumulation_steps == 0:
            do_step = True
        else:
            do_step = False
        self._propagate_loss(loss, do_step=do_step)
        if i % self.checkpoint_interval == 0 or i+1 == len(trainloader):
            self.save_checkpoint(i, epoch)
        if evalloader is not None and (i % self.eval_interval == 0 or i+1 == len(trainloader)):
            self._eval_sample(evalloader, step=i, epoch=epoch, sample_size=self.eval_sample_size)


    @abstractmethod
    def _train_step(self, batch, logging, epoch, step, *args, **kwargs):
        pass

    def _propagate_loss(self, loss, do_step=True):
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if do_step:
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

    def save_checkpoint(self, step, epoch):
        checkpoint = {
            "dims": self.model.dims,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "epoch": epoch,
        }
        f_ts = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"{self.checkpoint_dir}/{f_ts}_{self.trainer_name}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = f"{save_dir}/{f_ts}_{self.trainer_name}_whisperbrain_epoch-{epoch}_step-{step}.pt"
        torch.save(checkpoint, filepath)

class GANTrainer(BaseTrainer):
    """This is a base class for training a GAN model. It modifies some things from the Trainer class to accomodate the GAN training loop.

    Modifications:
    1. The __init__ method takes an additional critic argument which is the critic model.
    2. The __init__ method replaces the optimizer argument with a generator_optimizer argument and adds a critic_optimizer argument.
    3. The __init__ method replaces the scheduler argument with a generator_scheduler argument and adds a critic_scheduler argument.
    4. The train_epoch method is modified to train the critic and generator in alternating steps.
    5. The _train_step method is replaced with the _train_generator and _train_critic methods.
    """
    def __init__(
        self,
        trainer_name,
        model,
        generator_optimizer,
        critic,
        critic_optimizer,
        checkpoint_dir,
        checkpoint_interval=100,
        eval_interval=100,
        logging_interval=10,
        generator_scheduler=None,
        critic_scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1
        ):

        super().__init__(model, decoding_options, device=device, eval_sample_size=eval_sample_size)

        self.trainer_name = trainer_name

        self.critic = critic
        self.generator_optimizer = generator_optimizer
        self.critic_optimizer = critic_optimizer
        self.generator_scheduler = generator_scheduler
        self.critic_scheduler = critic_scheduler
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_interval = checkpoint_interval
        self.logging_interval = logging_interval
        self.eval_interval = eval_interval
        self.max_grad_norm = max_grad_norm
        self.gradient_accumulation_steps=gradient_accumulation_steps

    def train(self, trainloader, evalloader=None, epochs=1):
        self.model.train()
        for epoch in range(epochs):
            self.train_epoch(trainloader, evalloader, epoch)

    def train_epoch(self, trainloader, evalloader=None, epoch=0):
        evaluate = False
        for i, batch in enumerate(trainloader):
            self._step_epoch(i, batch, trainloader, evalloader, epoch)

    def _step_epoch(self, i, batch, trainloader, evalloader, epoch):
            logging = i % self.logging_interval == 0 or i+1 == len(trainloader)
            if evalloader is not None:
                evaluate = i % self.eval_interval == 0 or i+1 == len(trainloader)
            checkpoint = i % self.checkpoint_interval == 0 or i+1 == len(trainloader)
            d_loss = self._train_critic_step(batch, logging, epoch, i)
            if i % self.gradient_accumulation_steps == 0:
                do_step = True
            else:
                do_step = False
            self._propogate_critic_loss(d_loss, do_step=do_step)
            g_loss = self._train_generator_step(batch, logging, epoch, i)
            self._propogate_critic_loss(g_loss, do_step=do_step)
            if checkpoint:
                self.save_checkpoint(i, epoch)
            if evaluate:
                self._eval_sample(evalloader, step=i, epoch=epoch, sample_size=self.eval_sample_size)

    @abstractmethod
    def _train_critic_step(self, batch, logging, epoch, step):
        pass

    def _propogate_generator_loss(self, loss):
        loss.backward()
        if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.generator_optimizer.step()
        if self.generator_scheduler is not None:
            self.generator_scheduler.step()

    @abstractmethod
    def _train_generator_step(self, batch, logging, epoch, step):
        pass

    def _propogate_critic_loss(self, loss, do_step=True):
        loss.backward()
        if self.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        if do_step:
            self.critic_optimizer.step()
            if self.critic_scheduler is not None:
                self.critic_scheduler.step()

    def save_checkpoint(self, step, epoch):
        model_checkpoint = {
            "params": self.model.dims,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.generator_optimizer.state_dict(),
            "scheduler_state_dict": self.generator_scheduler.state_dict() if self.generator_scheduler is not None else None,
            "epoch": epoch,
        }
        critic_checkpoint = {
            "dims": self.critic.dims,
            "model_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.critic_optimizer.state_dict(),
            "scheduler_state_dict": self.critic_scheduler.state_dict() if self.critic_scheduler is not None else None,
            "epoch": epoch,
        }
        f_ts = self.timestamp.strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"{self.checkpoint_dir}/{f_ts}_{self.trainer_name}"
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        make_save_path = lambda model_type: f"{save_dir}/{f_ts}_{self.trainer_name}_{model_type}_epoch-{epoch}_step-{step}.pt"
        torch.save(model_checkpoint, make_save_path("whisperbrain"))
        torch.save(critic_checkpoint, make_save_path("critic"))


class TeacherStudentEmbeddingTrainer(Trainer):
    """This trainer aligns the embeddings of the pretrained AudioEncoder with the embeddings of the BrainEncoder.

    It uses masked MSE to align the embeddings on the part of the non-padded part of the sequence. Optionally, it can do a weighted sum of the unmasked and masked parts of the sequence to maintain stability when the transormer part of the brain_encoder is unlocked.

    This trainer is used to align the embeddings of the pretrained AudioEncoder with the embeddings of the BrainEncoder during two phases of training:
    1. When the BrainEncoder prenet is the only part of the BrainEncoder that is unlocked.
    2. When the entire BrainEncoder is unlocked.
    The AudioEncoder is frozen during both training phases. MSE is not a good loss for aligning two unlocked encoders.
    """
    def __init__(
            self,
            model,
            optimizer,
            checkpoint_dir,
            mask_weight=1,
            checkpoint_interval=100,
            logging_interval=10,
            eval_interval=100,
            scheduler=None,
            max_grad_norm=1,
            decoding_options=None,
            device=None,
            eval_sample_size=15,
            gradient_accumulation_steps=1):

        super().__init__(
                    "embedding-teacher-student",
                    model,
                    optimizer,
                    checkpoint_dir,
                    checkpoint_interval,
                    logging_interval,
                    eval_interval,
                    scheduler,
                    max_grad_norm,
                    decoding_options,
                    device,
                    eval_sample_size,
                    gradient_accumulation_steps
                )
        self.set_mask_weight(mask_weight)

    def set_mask_weight(self, mask_weight):
        if mask_weight < 0 or mask_weight > 1:
            raise ValueError(f"alpha should be between 0 and 1, but got {mask_weight}")
        self.mask_weight = mask_weight

    def _train_step(self, batch, step, epoch, logging):
        self.optimizer.zero_grad()
        self.model.train()
        brain_data, mels, _, sequence_lengths = batch
        brain_data.to(self.device)
        mels.to(self.device)
        # Forward pass for brain and audio
        brain_embeddings = self.model.embed_brain(brain_data)
        audio_embeddings = self.model.embed_audio(mels)
        if logging:
            log_dict = {
                "epoch": epoch,
                }
        if self.mask_weight == 1:
            loss = self._masked_sim(
                brain_embeddings,
                audio_embeddings,
                sequence_lengths)
            if logging:
                log_dict["masked_embedding_cos"] = loss
        elif self.mask_weight == 0:
            loss = self._unmasked_sim(
                brain_embeddings,
                audio_embeddings)
            if logging:
                log_dict["unmasked_embedding_cos"] = loss
        else:
            masked_loss = self._masked_sim(
                brain_embeddings,
                audio_embeddings,
                sequence_lengths)
            unmasked_loss = self._unmasked_sim(
                brain_embeddings,
                audio_embeddings)
            loss = self.mask_weight * masked_loss  +  (1 - self.mask_weight) * unmasked_loss
            if logging:
                log_dict["masked_embedding_cos"] = masked_loss
                log_dict["unmasked_embedding_cos"] = unmasked_loss
                log_dict["weighted_embedding_cos"] = loss

        if logging:
            wandb.log(log_dict)
        return loss

    def _masked_sim(self,
                    brain_embeddings,
                    audio_embeddings,
                    sequence_lengths):
        loss = 0
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        for i, sl in enumerate(sequence_lengths):
            x = brain_embeddings[i, :sl, :]
            y = audio_embeddings[i, :sl, :]
            loss += loss_fn(x, y, torch.ones(sl, device=x.device))
        return loss / len(sequence_lengths)

    def _unmasked_sim(self, brain_embeddings, audio_embeddings):
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        N, S, D = brain_embeddings.shape
        x = brain_embeddings.view(N*S, D)
        y = audio_embeddings.view(N*S, D)
        return loss_fn(x, y, torch.ones(N*S, device=x.device))


class EmbeddingGANTrainer(GANTrainer):
    """This trainer aligns the embeddings of the pretrained AudioEncoder with the embeddings of the BrainEncoder using adversarial training.

    Parameters
    ----------
    model: WhisperBrain
        The model to train
    critic: nn.Module
        The critic model to use for adversarial training
    optimizer: torch.optim.Optimizer
        The optimizer to use for training
    scheduler: torch.optim.lr_scheduler._LRScheduler or None
        The learning rate scheduler to use for training
    device: Union[str, torch.device]
        The device to train on
    """

    def __init__(
        self,
        model,
        generator_optimizer,
        critic,
        critic_optimizer,
        checkpoint_dir,
        loss_weights = None,
        checkpoint_interval=100,
        eval_interval=100,
        logging_interval=10,
        generator_scheduler=None,
        critic_scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1
        ):

        super().__init__(
            "embedding-gan",
            model=model,
            generator_optimizer=generator_optimizer,
            critic=critic,
            critic_optimizer=critic_optimizer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            eval_interval=eval_interval,
            logging_interval=logging_interval,
            generator_scheduler=generator_scheduler,
            critic_scheduler=critic_scheduler,
            max_grad_norm=max_grad_norm,
            decoding_options=decoding_options,
            device=device,
            eval_sample_size=eval_sample_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        self.weights = loss_weights


    def _train_critic_step(self, batch, logging, epoch, step):
        self.critic_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.model.eval()
        self.critic.train()

        brain_data, mels, _, _ = batch
        brain_data = brain_data.to(self.device)
        mels = mels.to(self.device)

        brain_embeddings = self.model.brain_encoder(brain_data).to(self.device)
        audio_embeddings = self.model.whisper.encoder(mels).to(self.device)

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
        logits = self.critic(mixed_embeddings).to(self.device)
        bce = torch.nn.BCELoss(reduction="mean").to(self.device)
        d_loss = bce(logits, ground_truth)

        if logging:
            log_dict = {
                "epoch": epoch,
                "embedding_gan_critic_loss": d_loss,
                }
            wandb.log(log_dict)

        return d_loss

    def _train_generator_step(self, batch, logging, epoch, step):
        self.generator_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.model.train()
        self.critic.train()

        brain_data, mels, _, signal_lengths = batch
        brain_data = brain_data.to(self.device)
        mels = mels.to(self.device)

        brain_embeddings = self.model.brain_encoder(brain_data)
        audio_embeddings = self.model.whisper.encoder(mels)

        # Pass brain embeddings through critic
        fake_logits = self.critic(brain_embeddings)

        # Generator loss (flip labels for the generator)
        bce = torch.nn.BCELoss(reduction="mean").to(self.device)
        g_loss = bce(fake_logits, torch.ones_like(fake_logits))

        sup_loss = self._supplemental_loss(brain_embeddings, audio_embeddings)
        weighted_loss = self.weights["g_loss"] * g_loss + self.weights["sup_loss"] * sup_loss
        if logging:
            wandb.log({
                    "embedding_gan_generator_loss": g_loss,
                    "embedding_gan_cosine_similarity": sup_loss,
                    "embedding_gan_weighted_loss": weighted_loss,
                    "epoch": epoch
                })
        return weighted_loss

    def _supplemental_loss(self, brain_embeddings, audio_embeddings):
        loss_fn = torch.nn.CosineEmbeddingLoss(reduction="mean")
        N, S, D = brain_embeddings.shape
        x = brain_embeddings.view(N, S*D)
        y = audio_embeddings.view(N, S*D)
        return loss_fn(x, y, torch.ones(N, device=x.device))


class TeacherStudentLogitTrainer(Trainer):
    """This trainer aligns the logits from passing the audio data through the AudioEncoder and
    Decoder with the logits from passing the brain data through the BrainEncoder and Decoder.

    Parameters
    ----------
    model: WhisperBrain
        The model to train
    optimizer: torch.optim.Optimizer
        The optimizer to use for training
    scheduler: torch.optim.lr_scheduler._LRScheduler or None
        The learning rate scheduler to use for training
    device: Union[str, torch.device]
        The device to train on
    """
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir,
        loss="kldiv",
        forcing_prob=0.5,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        normalizer=None,
        tokenizer=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1
        ):
        super().__init__(
            f"logit-{loss}-teacher-student",
            model,
            optimizer,
            checkpoint_dir,
            checkpoint_interval,
            logging_interval,
            eval_interval,
            scheduler,
            max_grad_norm,
            decoding_options,
            device,
            eval_sample_size,
            gradient_accumulation_steps
            )
        self.loss_options = {
            "kldiv": self._kl_divergence_loss,
            "wasserstein": self._wasserstein_loss,
            "mse": self._mse_loss
        }
        try:
            self.loss_fn = self.loss_options[loss]
            self.loss_fn_name = "logit_" + loss
        except KeyError:
            raise ValueError(f"loss should be one of {list(self.loss_options.keys())} but got {loss}")
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")
        self.forcing_prob = forcing_prob

    def _train_step(self, batch, step, epoch, logging):
        self.optimizer.zero_grad()
        self.model.train()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data)
        audio_embeddings = self.model.embed_audio(mels)
        tokens = self._encode_texts(texts)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        loss = 0
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
            loss += self.loss_fn(brain_logits, audio_logits) # note that for crossentropy we would use logits[:, -1, :] and tokens[:, t] but we want to get all the logits looking like the audio model for this training phase
            brain_generated_tokens += [torch.argmax(brain_logits[:, -1, :]).item()]
            audio_generated_tokens += [torch.argmax(audio_logits[:, -1, :]).item()]
        loss /= tokens.size(0) * tokens.size(1)
        if logging:
            wandb.log({
                "epoch": epoch,
                self.loss_fn_name: loss
                })
        return loss

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

    def _mse_loss(self, p, q):
        return torch.nn.functional.mse_loss(p, q)

    def _kl_divergence_loss(self, p, q):
        # logit shape (N, S, 51864)
        # apply kl divergence to each (51864) logit in each (N, S) sequence
        # return the batch mean (N,) of the kl divergence
        # p and q are the logits from the brain and audio encoders
        N, S, V = p.shape
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        divs = torch.sum(p * (torch.log(p) - torch.log(q)), dim=-1).mean(dim=0)
        weights = torch.arange(1, S+1, device=p.device).float() / S
        return torch.dot(divs, weights)

    def _wasserstein_loss(self, p, q):
        # logit shape (N, S, 51864)
        # flatten the batch and sequence dimensions
        p = F.softmax(p, dim=-1)
        q = F.softmax(q, dim=-1)
        # cumsums
        p_cumsum = torch.cumsum(p, dim=-1) # (N, S, 51864)
        q_cumsum = torch.cumsum(q, dim=-1)
        # wasserstein distance
        S = p.shape[1]
        dists = torch.mean(torch.abs(p_cumsum - q_cumsum), dim=-1).mean(dim=0)
        weights = torch.arange(1, S+1, device=p.device).float() / S
        return torch.dot(dists, weights)

class LogitGANTrainer(GANTrainer):
    def __init__(
        self,
        model,
        generator_optimizer,
        critic,
        critic_optimizer,
        checkpoint_dir,
        loss_weights = None,
        forcing_prob=0.5,
        checkpoint_interval=100,
        eval_interval=100,
        logging_interval=10,
        generator_scheduler=None,
        critic_scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        normalizer=None,
        tokenizer=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1
        ):

        super().__init__(
            "logit-gan",
            model=model,
            generator_optimizer=generator_optimizer,
            critic=critic,
            critic_optimizer=critic_optimizer,
            checkpoint_dir=checkpoint_dir,
            checkpoint_interval=checkpoint_interval,
            eval_interval=eval_interval,
            logging_interval=logging_interval,
            generator_scheduler=generator_scheduler,
            critic_scheduler=critic_scheduler,
            max_grad_norm=max_grad_norm,
            decoding_options=decoding_options,
            device=device,
            eval_sample_size=eval_sample_size,
            gradient_accumulation_steps=gradient_accumulation_steps
        )
        self.weights = loss_weights
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")
        self.forcing_prob = forcing_prob

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

    def _supplemental_loss(self, brain_logits, audio_logits):
        # kl divergence loss
        N, S, V = brain_logits.shape
        x = F.softmax(brain_logits.view(N*S, V), dim=-1)
        y = F.log_softmax(audio_logits.view(N*S, V), dim=-1)
        kldiv = torch.nn.KLDivLoss(reduction="batchmean")
        return kldiv(y, x)

    def _train_critic_step(self, batch, logging, epoch, step):
        self.critic_optimizer.zero_grad()
        self.generator_optimizer.zero_grad()
        self.model.eval()
        self.critic.train()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)
        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        d_loss = 0
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
            critic_logits = self.critic(mixed_logits)
            bce = torch.nn.BCELoss(reduction="mean").to(self.device)
            d_loss += bce(critic_logits, ground_truth)
        d_loss /= tokens.size(1)
        if logging:
            wandb.log({
                "epoch": epoch,
                "logit_gan_critic_loss": d_loss
                })
        return d_loss

    def _train_generator_step(self, batch, logging, epoch, step):
        self.generator_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        self.model.train()
        self.critic.train()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data).to(self.device)
        audio_embeddings = self.model.embed_audio(mels).to(self.device)
        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        g_loss = 0
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
            # Pass brain embeddings through critic
            N, S, V = brain_logits.shape
            critic_logits = self.critic(brain_logits)
            fake_truth = torch.ones((N, 1)).to(self.device)
            bce = torch.nn.BCELoss(reduction="mean").to(self.device)
            g_loss += bce(critic_logits, fake_truth)
        g_loss /= tokens.size(1)
        sup_loss = self._supplemental_loss(brain_logits, audio_logits)
        weighted_loss = self.weights["g_loss"] * g_loss + self.weights["sup_loss"] * sup_loss
        if logging:
            wandb.log({
                "epoch": epoch,
                "logit_gan_generator_loss": g_loss,
                "logit_gan_brain_audio_kldiv": sup_loss,
                "logit_gan_weighted_loss": weighted_loss
                })
        return weighted_loss



class EncoderDecoderTrainer(Trainer):
    """This trainer trains the WhisperBrain model using the cross entropy loss between the logits and the tokenized texts.

    Parameters
    ----------
    model: WhisperBrain
        The model to train
    optimizer: torch.optim.Optimizer
        The optimizer to use for training
    scheduler: torch.optim.lr_scheduler._LRScheduler or None
        The learning rate scheduler to use for training
    device: Union[str, torch.device]
        The device to train on
    """
    def __init__(
        self,
        model,
        optimizer,
        checkpoint_dir,
        forcing_prob=1,
        checkpoint_interval=100,
        logging_interval=10,
        eval_interval=100,
        scheduler=None,
        max_grad_norm=1,
        decoding_options=None,
        device=None,
        normalizer=None,
        tokenizer=None,
        eval_sample_size=15,
        gradient_accumulation_steps=1
        ):
        super().__init__(
            "encoder-decoder",
            model,
            optimizer,
            checkpoint_dir,
            checkpoint_interval,
            logging_interval,
            eval_interval,
            scheduler,
            max_grad_norm,
            decoding_options,
            device,
            eval_sample_size,
            gradient_accumulation_steps
            )
        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = EnglishTextNormalizer()
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = whisper.tokenizer.get_tokenizer("en")
        self.forcing_prob = forcing_prob

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

    def _train_step(self, batch, step, epoch, logging):
        self.optimizer.zero_grad()
        self.model.train()
        brain_data, mels, texts, _ = batch
        brain_data.to(self.device)
        mels.to(self.device)
        brain_embeddings = self.model.embed_brain(brain_data)
        tokens = self._encode_texts(texts).to(self.device)
        sots = list(self.tokenizer.sot_sequence_including_notimestamps)
        start_t = len(sots)
        loss = 0
        brain_generated_tokens = [sots.copy() for _ in range(len(texts))]
        for t in range(start_t, tokens.size(1)):
            if random.random() >= self.forcing_prob and start_t > len(sots):
                brain_input = torch.tensor([brain_generated_tokens], device=brain_embeddings.device)
            else:
                brain_input = tokens[:, :t+1]
            brain_logits = self.model.logits(brain_input, brain_embeddings)
            for i in range(len(texts)):
                x = brain_logits[i, :, :]
                y_index = tokens[i, :t+1]
                loss += torch.nn.CrossEntropyLoss()(x, y_index)
                brain_generated_tokens[i].append(torch.argmax(brain_logits[i, -1, :]).item())
        loss /= tokens.size(0) * tokens.size(1)
        if logging:
            wandb.log({
                "epoch": epoch,
                "sequence_crossentropy": loss
                })
        return loss
