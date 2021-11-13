# Métodos de Output Sampling na Geração de Música - PT

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Rquerimento de sistema

* Recomendado usar gerenciador de ambientes Miniconda
* Python 3.6.7
* Instalar os pacotes com o comando **pip**

E.g.

```
pip install -r requirements.txt
```
* Instalar cuda 8.0.61
* Instalar cudnn 8.0 v7.1

## Treinamento

No repositório já possui uma rede neural LSTM treinada para Bossa Nova com o filename **weights4.hdf5**. Caso queria gerar uma nova rede neural com novas músicas ou ajustes, deposite arquivos .midi na pasta **./midi_songs** e rode o comando abaixo:

E.g.

```
python lstm.py
```

**Observação**: Indicado colocar arquivos midis com apenas um tipo de Instrumento, também é possivel parar o processo de treinamento a qualquer momento, pois há uma callback que salva pesos da rede em cada época para retorno de treinamento ou para utilização do modelo em si.

## Inferência de Músicas

Com a rede neural treinada é possivel inferir novas amostras de músicas de 2 minutos ou 500 notas com 6 arquivos de **predict**

* predict_beam_search.py
* predict_greedy.py
* predict_random_temperature.py
* predict_temperature.py
* predict_top_k.py
* predict_top_p.py

E.g.

```
python predict_(O método de output selecionado).py
```

**Observação**: Alguns dos métodos listados possuem parâmetros, é indicado que entre no código e modifique esses parametros para testar a inferência.


# Music Generation with Output Sampling Decoders - EN

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

* Python 3.x
* Installing the following packages using pip:
	* Music21
	* Keras
	* Tensorflow
	* h5py

## Training

To train the network you run **lstm.py**.

E.g.

```
python lstm.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **predict.py**

E.g.

```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file
