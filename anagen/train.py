import torch
import logging
import tqdm
from anagen.dataset import collate

from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler

DEFAULT_TRAIN_BATCH_SIZE=1
logger = logging.getLogger(__name__)

# based on transformers/run_lm_finetuning
def train(args, train_dataset, model):
    train_sampler = SequentialSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler,
                                  batch_size=DEFAULT_TRAIN_BATCH_SIZE,
                                  collate_fn=collate)

    # not sure what the following does, copied from run_lm_finetuning code
    """
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps \
            // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // \
            args.gradient_accumulation_steps * args.num_train_epochs
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # start training
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size = %d", args.train_batch_size)

    model.zero_grad()

    # TODO: add train from checkpoint

    train_iterator = tqdm.trange(int(args.num_train_epochs), desc="Epoch")

    # TODO: set seed

    for _ in train_iterator:
        # epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(train_dataloader):
            # TODO: transfer to device
            inputs = batch
            return inputs

            # model.train()
            # outputs = model(inputs)
            # loss = outputs["loss"]
            # loss.backward()
