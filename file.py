
from absl import app, flags, logging
# import IPython
import torch as th
import pytorch_lightning as pl

import nlp 
import transformers

# sh.rm('-r', '-f', 'logs')

flags.DEFINE_bool('debug', True, '')
flags.DEFINE_integer('epochs', 10, '')
flags.DEFINE_integer('batch_size', 8, '')
flags.DEFINE_float('momentum', 0.9, '')
flags.DEFINE_float('lr', 0.01, '')
flags.DEFINE_string('model',"bert-base-uncased", '')
flags.DEFINE_integer('seq_length', 32, '')
FLAGS = flags.FLAGS

class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction = 'none')
    def prepare_data(self):
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
        
        def tokenize(x):
            x['input_ids'] = tokenizer.encode(x['text'], 
                                              max_length = FLAGS.seq_length, 
                                              pad_to_max_length = True)
            
            return x
        
        def prepare_ds(split):
            dataset = nlp.load_dataset('imdb', split =f'{split}[:{FLAGS.batch_size if FLAGS.debug else "5%"}]')
            dataset = dataset.map(tokenize)
            dataset.set_format(type = 'torch', output_all_column = ['input_ids', 'label'])
            return dataset
        
        self.train_ds, self.test_ds = map(prepare_ds, ('train', 'test'))
        return  
        
    def forward(self, input_ids):
        mask = (input_ids !=0).float()
        logits, = self.model(input_ids, mask)
        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        return {'loss':loss}
    
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['input_ids'])
        loss = self.loss(logits, batch['label'])
        accuracy = (logits.argmax(-1) == batch['label']).float()
        return {'loss':loss, 'accuracy':accuracy}
    
    def validation_epoch_ends(self, outputs):
        losses = th.cat([output['loss'] for output in outputs], 0).mean()
        accuracy = th.cat([output['accuracy'] for output in outputs], 0)
        out = {'loss':losses, 'accuracy':accuracy}
        return {**out, 'log:': out}
    
    def train_dataloader(self):
        return th.utils.data.DataLoader(self.train_ds, batch_size = FLAGS.batch_size, drop_last = True, shuffle = True,)

    def val_dataloader(self):
        return th.utils.data.DataLoader(self.test_ds, batch_size = FLAGS.batch_size, drop_last = False, shuffle = True,)
    
    def configure_optimizers(self):
        optimizer = th.optim.SGD(self.parameters(), lr = FLAGS.lr, momentum = FLAGS.momentum)
        return optimizer
    
def main(_):
    logging.info ("Hello World")
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(default_root_dir ='logs',
                         gpus = 0,
                         max_epochs = FLAGS.epochs,
                         fast_dev_run = FLAGS.debug,
                         logger = pl.loggers.TensorBoardLogger('logs/', name = 'imdb', version = 0)
                         )
    trainer.fit(model)
    
if __name__=='__main__':
    app.run(main)