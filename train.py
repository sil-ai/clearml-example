import time

import clearml
from clearml import Task
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import default_data_collator
from transformers import AutoModelForQuestionAnswering, TrainingArguments, Trainer


def preprocess_function(examples):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        answer = answers[i]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label it (0, 0)
        if offset[context_start][0] > end_char or offset[context_end][1] < start_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


if __name__ == '__main__':

    # ClearML stuff
    Task.add_requirements("-rrequirements.txt")
    task = Task.init(
      project_name='IDX_AdAPT_Experiments',    # project name of at least 3 characters
      task_name='test-demo-' + str(int(time.time())), # task name of at least 3 characters
      task_type="training",
      tags=None,
      reuse_last_task_id=True,
      continue_last_task=False,
      output_uri="s3://adapt-experiments",
      auto_connect_arg_parser=True,
      auto_connect_frameworks=True,
      auto_resource_monitoring=True,
      auto_connect_streams=True,    
    )
    
    # Load and pre-process squad dataset
    squad = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_squad = squad.map(preprocess_function, batched=True, remove_columns=squad["train"].column_names)
    data_collator = default_data_collator

    # Load the model that we will pre-train
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")

    # Training arguments
    training_args = TrainingArguments(
      output_dir='./results',
      evaluation_strategy="epoch",
      learning_rate=2e-5,
      per_device_train_batch_size=16,
      per_device_eval_batch_size=16,
      num_train_epochs=0.1,
      weight_decay=0.01,
    )

    # Trainer setup
    trainer = Trainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_squad["train"],
      eval_dataset=tokenized_squad["validation"],
      data_collator=data_collator,
      tokenizer=tokenizer,
    )

    # Train the model
    trainer.train()

    # Save the model
    trainer.save_model("squad-ex")

    # Save the artifact in ClearML
    task.upload_artifact(name='squad-ex', artifact_object='squad-ex/')


