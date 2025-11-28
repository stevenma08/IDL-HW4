from .base_trainer import BaseTrainer
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Dict, Tuple, Any, Optional, List
from ..utils import create_scheduler
from ..decoding.sequence_generator import SequenceGenerator

class LMTrainer(BaseTrainer):
    """
    Language Model Trainer class that handles the training, validation, and generation loops.
    """

    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        
        # Robustly get config parameters with defaults to prevent crashes
        smoothing_val = self.config.get('loss', {}).get('label_smoothing', 0.0)
        pad_id = self.tokenizer.pad_id

        # Initialize CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(
            ignore_index=pad_id,
            label_smoothing=smoothing_val
        )

    def _train_epoch(self, dataloader) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """
        Train for one epoch.
        """
        self.model.train()
        
        # Setup progress bar
        pbar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Training LM]")
        
        total_loss = 0.0
        token_count = 0
        
        # Ensure gradients are zeroed before starting
        self.optimizer.zero_grad()

        # Iterate through batch
        for step, (inputs, targets, lengths) in enumerate(dataloader):
            
            # 1. Move data to device
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            # 2. Forward pass with Mixed Precision
            with torch.autocast(device_type=self.device, dtype=torch.float16):
                # Forward pass through the model
                logits, attn_maps = self.model(inputs, lengths)

                # Calculate Loss
                # Reshape logits to (Batch * Seq, Vocab) and targets to (Batch * Seq) inline
                loss_val = self.criterion(
                    logits.view(-1, self.model.num_classes), 
                    targets.view(-1)
                )
                
            # 3. Track metrics (using item() to detach from graph)
            current_tokens = lengths.sum().item()
            token_count += current_tokens
            total_loss += loss_val.item() * current_tokens

            # 4. Backward pass with Gradient Accumulation
            # Scale the loss by accumulation steps
            accum_steps = self.config['training']['gradient_accumulation_steps']
            scaled_loss = loss_val / accum_steps
            
            self.scaler.scale(scaled_loss).backward()
        
            # 5. Optimizer Step (only every accum_steps)
            if (step + 1) % accum_steps == 0:
                self.scaler.step(self.optimizer)
                
                # Step the scheduler (unless it's Plateua based)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                
                self.scaler.update()
                self.optimizer.zero_grad()

            # 6. Update Progress Bar
            avg_loss = total_loss / token_count
            perplexity = torch.exp(torch.tensor(avg_loss))
            
            pbar.set_postfix(
                ce_loss_token=f"{avg_loss:.4f}",
                perplexity_token=f"{perplexity:.4f}",
                acc_step=f"{(step % accum_steps) + 1}/{accum_steps}"
            )
            pbar.update()

            # Memory cleanup
            del inputs, targets, lengths, logits, loss_val
            torch.cuda.empty_cache()

        # Handle any remaining gradients if dataset size isn't divisible by accum_steps
        if (len(dataloader) % accum_steps) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        # Final Epoch Metrics
        final_loss = total_loss / token_count
        final_loss_char = final_loss / dataloader.dataset.get_avg_chars_per_token()
        
        final_ppl = torch.exp(torch.tensor(final_loss)).item()
        final_ppl_char = torch.exp(torch.tensor(final_loss_char)).item()
        
        pbar.close()

        return {
            'ce_loss_token': final_loss,
            'ce_loss_char': final_loss_char,
            'perplexity_token': final_ppl,
            'perplexity_char': final_ppl_char
        }, attn_maps
            
            
    def _validate_epoch(self, dataloader):
        """
        Validate for one epoch.
        """
        self.model.eval()
        pbar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Validating LM]")
        
        total_loss = 0.0
        token_count = 0
        attn_maps = {}

        for step, (inputs, targets, lengths) in enumerate(dataloader):
            # Move to device
            inputs  = inputs.to(self.device)
            targets = targets.to(self.device)
            lengths = lengths.to(self.device)

            # Inference Mode (No gradients)
            with torch.inference_mode():
                # Forward pass
                logits, attn_maps = self.model(inputs, lengths)

                # Loss calculation
                loss_val = self.criterion(
                    logits.view(-1, self.model.num_classes),
                    targets.view(-1)
                )

            # Metric tracking
            current_tokens = lengths.sum().item()
            token_count += current_tokens
            total_loss += loss_val.item() * current_tokens

            # Update Pbar
            avg_loss = total_loss / token_count
            perplexity = torch.exp(torch.tensor(avg_loss))
            pbar.set_postfix(
                ce_loss_token=f"{avg_loss:.4f}",
                perplexity_token=f"{perplexity:.4f}",
            )
            pbar.update()

            del inputs, targets, lengths, logits, loss_val
            torch.cuda.empty_cache()

        # Final Metrics
        final_loss = total_loss / token_count
        final_loss_char = final_loss / dataloader.dataset.get_avg_chars_per_token()
        final_ppl = torch.exp(torch.tensor(final_loss)).item()
        final_ppl_char = torch.exp(torch.tensor(final_loss_char)).item()
        
        pbar.close()

        return {
            'ce_loss_token': final_loss,
            'ce_loss_char': final_loss_char,
            'perplexity_token': final_ppl,
            'perplexity_char': final_ppl_char
        }, attn_maps
        

    def train(self, train_dataloader, val_dataloader, epochs: int):
        """
        Full training loop.
        """
        if self.scheduler is None: raise ValueError("Scheduler not initialized")
        if self.optimizer is None: raise ValueError("Optimizer not initialized")
        
        best_metric = float('inf')

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            
            # Execute Training and Validation
            train_stats, train_attn = self._train_epoch(train_dataloader)
            val_stats, val_attn = self._validate_epoch(val_dataloader)
            
            # Execute Generation
            gen_outputs = self.generate(val_dataloader)
            
            # Scheduler Step (for Plateau)
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_stats['ce_loss_char'])

            # Logging
            self._log_metrics({'train': train_stats, 'val': val_stats}, epoch)
            
            # Save Visualizations
            if len(train_attn) > 0:
                k = list(train_attn.keys())[0]
                self._save_attention_plot(train_attn[k][0], epoch, "train_self")
            
            if len(val_attn) > 0:
                k = list(val_attn.keys())[0]
                self._save_attention_plot(val_attn[k][0], epoch, "val_self")

            # Save Text and Checkpoints
            self._save_generated_text(gen_outputs, f'val_epoch_{epoch}')
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Save Best Model
            current_metric = val_stats['ce_loss_char']
            if current_metric < best_metric:
                best_metric = current_metric
                self.best_metric = current_metric
                self.save_checkpoint('checkpoint-best-metric-model.pth')

            self.current_epoch += 1


    def evaluate(self, test_dataloader):
        """
        Evaluate on test set.
        """
        metrics, attention = self._validate_epoch(test_dataloader)
        self._log_metrics({'test': metrics}, self.current_epoch)  

        if len(attention) > 0:
            k = list(attention.keys())[0]
            self._save_attention_plot(attention[k][0], self.current_epoch, "test_self")

        gen_results = {}
        configs = self._get_evaluation_generation_configs()
        
        for name, cfg in configs.items():
            try:
                res = self.generate(test_dataloader, generation_config=cfg)
                gen_results[name] = res
                self._save_generated_text(res, f'test_epoch_{self.current_epoch}_{name}')
            except Exception as e:
                print(f"Error generating {name}: {e}")
                
        return metrics, gen_results

    def generate(self, dataloader, generation_config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Generate text from prompts.
        """
        # Default config if none provided
        if generation_config is None:
            generation_config = {
                'num_samples': 10, 'prompt_length': 20, 'seed': 11785,
                'max_length': self.model.max_len, 'temperature': 1.0,
                'beam_width': 1, 'repeat_penalty': 1.0, 'top_k': 0, 'top_p': 0.0    
            }

        # Initialize Generator
        gen_engine = SequenceGenerator(
            score_fn=lambda x: self.model.score(x),
            tokenizer=self.tokenizer,
            max_length=self.model.max_len,
            device=self.device
        )

        # Get Prompts
        prompts, original_seqs = dataloader.dataset.sample_prompts(
            num_samples=generation_config.get('num_samples', 10),
            prompt_length=generation_config.get('prompt_length', 10),
            seed=generation_config.get('seed', 11785)
        )
        prompts = prompts.to(self.device)

        self.model.eval()
        with torch.inference_mode():
            # Determine generation method based on config parameters
            if generation_config.get('top_k', 0) > 0 or generation_config.get('top_p', 0) > 0:
                print("Running Sampling...")
                out_seqs, _ = gen_engine.generate_sample(
                    prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    top_k=generation_config.get('top_k', 0),
                    top_p=generation_config.get('top_p', 0.0)
                )
            elif generation_config.get('beam_width', 1) > 1:
                print("Running Beam Search...")
                out_seqs, _ = gen_engine.generate_beam(
                    prompts,
                    beam_width=generation_config.get('beam_width', 1),
                    temperature=generation_config.get('temperature', 1.0),
                    repeat_penalty=generation_config.get('repeat_penalty', 1.0)
                )
                # Select best beam
                out_seqs = out_seqs[:, 0]
            else:
                print("Running Greedy Search...")
                out_seqs, scores = gen_engine.generate_greedy(
                    prompts,
                    temperature=generation_config.get('temperature', 1.0),
                    repeat_penalty=generation_config.get('repeat_penalty', 1.0)
                )

        # Truncate at EOS
        clean_seqs = gen_engine.post_process_sequence(out_seqs, self.tokenizer)

        # Format output
        output_list = []
        for i, (prompt, generated, original) in enumerate(zip(prompts, clean_seqs, original_seqs)):
            # Decode sequences
            prompt_str = self.tokenizer.decode(prompt.tolist())
            gen_str = self.tokenizer.decode(generated[len(prompt):].tolist())
            orig_str = self.tokenizer.decode(original[len(prompt):].tolist())
            
            output_list.append({
                'prompt': prompt_str,
                'original': orig_str,
                'generated': gen_str,
                'score': 0.0 # Placeholder
            })

        return output_list

    def _get_evaluation_generation_configs(self) -> Dict[str, Dict[str, Any]]:
        base_cfg = {'num_samples': 50, 'prompt_length': 10, 'seed': 11785, 'max_length': self.model.max_len}
        
        return {
            'greedy': {**base_cfg, 'temperature': 1.0, 'beam_width': 1, 'repeat_penalty': 1.0, 'top_k': 0, 'top_p': 0.0},
            'beam':   {**base_cfg, 'temperature': 1.0, 'beam_width': 10, 'repeat_penalty': 1.2, 'top_k': 0, 'top_p': 0.0},
            'sample': {**base_cfg, 'temperature': 1.0, 'beam_width': 1, 'repeat_penalty': 1.0, 'top_k': 10, 'top_p': 0.95}
        }