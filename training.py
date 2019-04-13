import numpy as np
from matplotlib import pyplot as plt
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from tests import test_prediction, test_generation
import random



# load all that we need

dataset = np.load('../dataset/wiki.train.npy')
fixtures_pred = np.load('../fixtures/prediction.npz')  # dev
fixtures_gen = np.load('../fixtures/generation.npy')  # dev
fixtures_pred_test = np.load('../fixtures/prediction_test.npz')  # test
fixtures_gen_test = np.load('../fixtures/generation_test.npy')  # test
vocab = np.load('../dataset/vocab.npy')



# data loader

device = "cuda" if torch.cuda.is_available() else "cpu"

class LanguageModelDataLoader(DataLoader):

    def __init__(self, dataset, batch_size, shuffle=True):
        
        # self.dataset=dataset[0:1]
        self.dataset = dataset
        self.batch_size=batch_size
        self.shuffle=shuffle
        self.num_workers=16


    def __iter__(self):

        #shuffle all sequences
        np.random.shuffle(self.dataset)

        # concatenate your articles and build into batches
        all_together=np.concatenate(self.dataset)

        n_seq=all_together.shape[0] // (SEQ_LEN*BATCH_SIZE)

        all_together=all_together[:n_seq*SEQ_LEN*BATCH_SIZE]

        batch_data=all_together.reshape(-1,BATCH_SIZE,SEQ_LEN)



        for n in range(batch_data.shape[0]):
            txt=batch_data[n]

            cut = random.randint(10, SEQ_LEN-10)


            yield (torch.from_numpy(txt[:,:cut-1]),torch.from_numpy(txt[:,1:cut]))

            yield (torch.from_numpy(txt[:, cut-1:-2]), torch.from_numpy(txt[:, cut:-1]))





class LanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super(LanguageModel, self).__init__()
        
        self.vacab_size=vocab_size
        self.embed_size=EMBED_SIZE
        self.hidden_size=HIDDEN_SIZE
        self.nlayers=NLAYERS

        self.embedding=nn.Embedding(self.vacab_size,self.embed_size)
        self.rnn=nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.nlayers)
        self.score=nn.Linear(self.hidden_size,vocab_size)


    def forward(self, x):
        batch_size=x.size(1)


        x=x.type(torch.int64)
        embed=self.embedding(x)
        hidden=None

        embed=embed.permute(1,0,2)

        output_lstm, hidden=self.rnn(embed,hidden)

        output_lstm_flatten=output_lstm.view(-1,self.hidden_size)
        output_flatten=self.score(output_lstm_flatten)

        return_value=output_flatten.view(-1,batch_size,self.vacab_size)

        return return_value

    def prediction(self,inp,is_generate=False,n_words=0):
        # inp=inp.T
        inp=torch.from_numpy(inp)
        inp=inp.to(device)

        generated_words=[]
        inp=inp.type(torch.int64)
        # embed=self.embedding(inp).unsqueeze(1)
        embed = self.embedding(inp)
        hidden=None
        embed=embed.permute(1,0,2)
        output_lstm,hidden=self.rnn(embed,hidden)
        output=output_lstm[-1]
        scores = self.score(output)

        if is_generate==False:
            scores=scores.to('cpu')
            return scores.detach().numpy()


        _,current_word=torch.max(scores,dim=1)
        generated_words.append(current_word.to('cpu').numpy())

        if n_words>1:
            for i in range(n_words-1):
                embed=self.embedding(current_word).unsqueeze(0)
                output_lstm,hidden=self.rnn(embed,hidden)
                output=output_lstm[0]
                scores=self.score(output)
                _,current_word=torch.max(scores,dim=1)
                generated_words.append(current_word.to('cpu').numpy())

        result_numpy=np.array(generated_words).T

        return result_numpy




# model trainer
class LanguageModelTrainer:
    def __init__(self, model, loader, max_epochs=1, run_id='exp'):

        self.model = model
        self.loader = loader
        self.train_losses = []
        self.val_losses = []
        self.predictions = []
        self.predictions_test = []
        self.generated_logits = []
        self.generated = []
        self.generated_logits_test = []
        self.generated_test = []
        self.epochs = 0
        self.max_epochs = max_epochs
        self.run_id = run_id

        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=0.001,weight_decay=1e-6)
        self.criterion = nn.CrossEntropyLoss().to(device)

    def train(self):
        self.model.train() # set to training mode
        self.model.to(device)
        epoch_loss = 0
        for batch_num, (inputs, targets) in enumerate(self.loader):
            inputs=inputs.to(device)
            targets=targets.to(device)
            epoch_loss += self.train_batch(inputs, targets)


        epoch_loss = epoch_loss / (batch_num + 1)
        self.epochs += 1
        print('[TRAIN]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, epoch_loss))
        self.train_losses.append(epoch_loss)

    def train_batch(self, inputs, targets):
        """ 
            Define code for training a single batch of inputs
        """
        outputs=model(inputs)
        loss=self.criterion(outputs.view(-1,outputs.size(2)),targets.flatten().type(torch.int64))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


        return loss

    
    def test(self):
        self.model.eval() # set to eval mode
        self.model=self.model.to(device)
        predictions = TestLanguageModel.prediction(fixtures_pred['inp'], self.model) # get predictions
        self.predictions.append(predictions)
        generated_logits = TestLanguageModel.generation(fixtures_gen, 10, self.model) # generated predictions for 10 words
        generated_logits_test = TestLanguageModel.generation(fixtures_gen_test, 10, self.model)
        nll = test_prediction(predictions, fixtures_pred['out'])
        generated = test_generation(fixtures_gen, generated_logits, vocab)
        generated_test = test_generation(fixtures_gen_test, generated_logits_test, vocab)
        self.val_losses.append(nll)
        
        self.generated.append(generated)
        self.generated_test.append(generated_test)
        self.generated_logits.append(generated_logits)
        self.generated_logits_test.append(generated_logits_test)
        
        # generate predictions for test data
        predictions_test = TestLanguageModel.prediction(fixtures_pred_test['inp'], self.model) # get predictions
        self.predictions_test.append(predictions_test)
            
        print('[VAL]  Epoch [%d/%d]   Loss: %.4f'
                      % (self.epochs + 1, self.max_epochs, nll))
        return nll

    def save(self):
        # don't change these
        model_path = os.path.join('experiments', self.run_id, 'model-{}.pkl'.format(self.epochs))
        torch.save({'state_dict': self.model.state_dict()},
            model_path)
        np.save(os.path.join('experiments', self.run_id, 'predictions-{}.npy'.format(self.epochs)), self.predictions[-1])
        np.save(os.path.join('experiments', self.run_id, 'predictions-test-{}.npy'.format(self.epochs)), self.predictions_test[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-{}.npy'.format(self.epochs)), self.generated_logits[-1])
        np.save(os.path.join('experiments', self.run_id, 'generated_logits-test-{}.npy'.format(self.epochs)), self.generated_logits_test[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated[-1])
        with open(os.path.join('experiments', self.run_id, 'generated-{}-test.txt'.format(self.epochs)), 'w') as fw:
            fw.write(self.generated_test[-1])


# In[ ]:


class TestLanguageModel:
    def prediction(inp, model):
        """
            :param inp:
            :return: a np.ndarray of logits
        """
        prediction_result=model.prediction(inp,is_generate=False)
        return prediction_result


        
    def generation(inp, forward, model):
        """
            TODO: write generation code here

            Generate a sequence of words given a starting sequence.
            :param inp: Initial sequence of words (batch size, length)
            :param forward: number of additional words to generate
            :return: generated words (batch size, forward)
        """        
        generation_result=model.prediction(inp,is_generate=True,n_words=10)

        return generation_result
        




# define other hyperparameters

NUM_EPOCHS = 20
BATCH_SIZE = 64
SEQ_LEN=100
EMBED_SIZE=512
HIDDEN_SIZE=128
NLAYERS=3
PRINT_CUT=100



run_id = str(int(time.time()))
if not os.path.exists('./experiments'):
    os.mkdir('./experiments')
os.mkdir('./experiments/%s' % run_id)
print("Saving models, predictions, and generated words to ./experiments/%s" % run_id)



model = LanguageModel(len(vocab))
loader = LanguageModelDataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True)
trainer = LanguageModelTrainer(model=model, loader=loader, max_epochs=NUM_EPOCHS, run_id=run_id)


best_nll = 1e30 
for epoch in range(NUM_EPOCHS):
    trainer.train()
    nll = trainer.test()
    if nll < best_nll:
        best_nll = nll
        print("Saving model, predictions and generated output for epoch "+str(epoch)+" with NLL: "+ str(best_nll))
        trainer.save()
    



# plot training curves
plt.figure()
plt.plot(range(1, trainer.epochs + 1), trainer.train_losses, label='Training losses')
plt.plot(range(1, trainer.epochs + 1), trainer.val_losses, label='Validation losses')
plt.xlabel('Epochs')
plt.ylabel('NLL')
plt.legend()
plt.show()




# see generated output
print (trainer.generated[-1]) # get last generated output






