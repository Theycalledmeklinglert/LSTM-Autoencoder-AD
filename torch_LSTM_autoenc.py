import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(LSTMAutoEncoder, self).__init__()
        self.device = device
        self.encoder = Encoder(num_layers, hidden_size, nb_feature, dropout=dropout, device=device)
        self.decoder = Decoder(num_layers, hidden_size, nb_feature, dropout=dropout, device=device)
        self.is_training = None

    def forward(self, input_seq):

        #print(f"input_seq shape: {input_seq.cpu().detach().numpy().shape}")
        #print(f"input_seq: {input_seq.cpu().detach().numpy()}")

        output = torch.ones(size=input_seq.shape, dtype=torch.float)
        hidden_cell = self.encoder(input_seq)                       # encoded hidden state of input seq
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2]) #takes last timestamp of input seq and reshaped it into (batch_size, 1 (timestep), number_of_features)

        #print(f"input_decoder shape: {input_decoder.cpu().detach().numpy().shape}")
        #print(f"input_decoder: {input_decoder.cpu().detach().numpy()}")

        # todo: missing linear layer for first decoder prediction value (see below)
        #if not self.is_training:
            #input_decoder = self.decoder.linear(hidden_cell) #[0])
            #print(f"input_decoder of linear shape before reshape: {input_decoder.cpu().detach().numpy().shape}")
            #print(f"input_decoder  of linear: {input_decoder.cpu().detach().numpy()}")
            #input_decoder = input_decoder[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])
            #print(f"input_decoder of linear shape after reshape: {input_decoder.cpu().detach().numpy().shape}")

        #print(f"outer input_decoder of linear shape after reshape: {input_decoder.cpu().detach().numpy().shape}")


        for i in range(input_seq.shape[1] - 1, -1, -1):                                     #go through input seq backwards
            output_decoder, hidden_cell = self.decoder(input_decoder, hidden_cell)  # hidden_cell[0] doesnt work
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]       # likely selects the first (and only) time step from the decoder output for all batches and all features.

        return output

            #print(f"output_decoder_shape: {output_decoder.cpu().detach().numpy().shape}")

            # if self.is_training:
            #     input_decoder = input_seq[:, i, :].view(input_seq.shape[0], 1, input_seq.shape[2])  #this should
            #     #print(f"acc_paper_input_decoder_shape: {input_decoder.cpu().detach().numpy().shape}")
            # else:
            #     input_decoder = output_decoder

            # Print the output_decoder for each loop iteration
            #print(f"Iteration {i}, output_decoder: {output_decoder.cpu().detach().numpy()}")  # Move to CPU and detach from graph for printing


#TODO: Fehlt hier nicht die Unterscheidung zwischen Training und Nicht training loop aus dem Paper?

class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Encoder, self).__init__()

        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
        self.initialize_hidden = True

    def initHidden(self, batch_size):
         self.hidden_cell = (
             torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device),
             torch.randn((self.num_layers, batch_size, self.hidden_size), dtype=torch.float).to(self.device)
         )

    def forward(self, input_seq):
        # if self.initialize_hidden:
        #     self.initHidden(input_seq.shape[0]) #todo: i think this is indeed correct
        #     self.initialize_hidden = False

        self.initHidden(input_seq.shape[0])
        _, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)   # , self.hidden_cell)
        return self.hidden_cell


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Decoder, self).__init__()

        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
        #self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)
        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)   #todo: Change!!!!!!

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        #output = self.linear(hidden_cell[0]) #[0])        #todo:     change    output = self.linear(output)
        output = self.linear(output)

        return output, hidden_cell




