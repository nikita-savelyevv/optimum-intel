import nncf
import numpy as np
import logging
import tempfile

from transformers import AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments, default_data_collator
from datasets import load_dataset
import evaluate
from optimum.intel import OVModelForSequenceClassification, OVConfig, OVTrainer


nncf.nncf_logger.setLevel(logging.ERROR)


def reshape_model(model):
    shapes = {}
    for inputs in model.inputs:
        shapes[inputs] = inputs.get_partial_shape()
        shapes[inputs][0] = -1
        shapes[inputs][1] = -1
    model.reshape(shapes)


def get_num_fqs(model):
    num_fake_quantize = 0
    for node in model.get_ops():
        if "FakeQuantize" in node.get_type_name():
            num_fake_quantize += 1
    return num_fake_quantize


model_id = "distilbert-base-uncased"
for _ in range(5):
    # n_samples = 16
    n_samples = (np.random.randint(1000) % 16) + 1

    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    ov_config = OVConfig()
    dataset = load_dataset("glue", "sst2")
    dataset = dataset.map(
        lambda examples: tokenizer(examples["sentence"], padding="max_length", max_length=128), batched=True
    )
    train_dataset = dataset["train"].select(range(n_samples))
    eval_dataset = dataset["validation"].select(range(n_samples))
    metric = evaluate.load("glue", "sst2")
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = OVTrainer(
            model=model,
            ov_config=ov_config,
            task="sequence-classification",
            args=TrainingArguments(tmp_dir, num_train_epochs=1.0, do_train=True, do_eval=True),
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=lambda p: metric.compute(predictions=np.argmax(p.predictions, 1), references=p.label_ids),
            tokenizer=tokenizer,
            data_collator=default_data_collator,
        )
        trainer.train()
        trainer.evaluate()
        trainer.save_model()

        ov_model = OVModelForSequenceClassification.from_pretrained(tmp_dir)
        fqs_before_reshape = get_num_fqs(ov_model.model)
        reshape_model(ov_model.model)
        fqs_after_reshape = get_num_fqs(ov_model.model)
        print(f"Number of FQ nodes before reshape: {fqs_before_reshape}, after reshape: {fqs_after_reshape}")
