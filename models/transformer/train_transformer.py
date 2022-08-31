from transformers import Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainer

from models.transformer.trainer_seq2seq_custom import Seq2SeqTrainerCustom
from utils.evaluation_metrics import Metrics


def train_transformer(train_data, val_data, tokenizer, config):
    training_args = Seq2SeqTrainingArguments(
        output_dir='./results',
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        logging_dir='./logs',
        logging_steps=10,
        warmup_steps=config.warmup_steps,
        load_best_model_at_end=True,
        predict_with_generate=True,
        generation_max_length=config.generation_max_length,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        report_to="wandb",
        run_name=f'{config.model_name.replace("/", "-")}-{config.num_epochs}epochs-{config.warmup_steps}'
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    metrics = Metrics(['exact_match', 'google_bleu', 'meteor', 'edit_distance'], tokenizer)

    trainer = Seq2SeqTrainerCustom(
        length_multiplier=config.length_multiplier,
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=metrics.compute_metrics
    )

    trainer.train()

    if config.save_model:
        trainer.save_model(f'{config.models_path}/{config.model_name.replace("/", "-")}-{config.num_epochs}epochs-{config.warmup_steps}')
