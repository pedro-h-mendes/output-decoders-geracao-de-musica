# Métodos de Output Sampling na Geração de Música - PT

Esse sistema é resultado de um TCC da Pós de Ciência de Dados e Machine Learning, possui o objeivo de testar inferencias com diferentes métodos de output sampling conhecidos no campo de PLN (Processamento de Linguagem Natual): Beam Search, Top K, Top P, Random Sampling, Greedy Sampling e Temperature Index.

Mais detalhes da pesquisa podem ser acessados nesse [Link](http://a.com) com o artigo para download.

Base de dados das músicas utilizadas no treinamento podem ser acessados e baixados nesse [Link](https://www.kaggle.com/macchi57/bossa-nova-midi). 

Musicas geradas na pesquisa estão complidadas nesse [Link](https://soundcloud.com/pedro-mendes-116/sets/metodos-de-output-sampling-na-geracao-de-musica-em-redes-neurais).

Qualquer dívidas sobre o trabalho podem ser enviadas no email: pedro.mendes@sempreceub.com

Ficarei feliz em responde-las.

Boas inferências!

## Requerimentos de sistema

* Recomendado usar gerenciador de ambientes Miniconda
* Python 3.6.7
* Instalar os pacotes com o comando **pip**:
	```
	pip install -r requirements.txt
	```
* Instalar cuda 8.0.61
* Instalar cudnn 8.0 v7.1

## Treinamento

No repositório já possui uma rede neural LSTM treinada para Bossa Nova com o filename **weights4.hdf5**. Caso queria gerar uma nova rede neural com novas músicas ou ajustes, deposite os arquivos .midi quistos na pasta **./midi_songs** e rode o comando abaixo:

```
python lstm.py
```

**Observação**: Indicado colocar arquivos midis com apenas um tipo de Instrumento, também sendo possivel parar o processo de treinamento a qualquer momento com segurança, pois há uma callback que salva pesos da rede em cada época para retorno de treinamento ou para utilização do modelo em si.

## Inferência de Músicas

Com a rede neural treinada é possivel inferir novas amostras de músicas de 2 minutos ou 500 notas com 6 possíveis arquivos de **predict**

* predict_beam_search.py
* predict_greedy.py
* predict_random_temperature.py
* predict_temperature.py
* predict_top_k.py
* predict_top_p.py

Rodar esse comando:

```
python predict_(O método de output selecionado).py
```

**Observação**: Alguns dos métodos listados possuem hiperparâmetros específicos, é indicado que entre nos códigos de **predict** e modifique esses parametros para testar a inferência.

## Referências

Esse sistema foi baseado na abordagem de [Sigurður Skúli](https://medium.com/@sigurdurssigurg) no artigo Medium [How to Generate Music using a LSTM Neural Network in Keras](https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5). Os códigos originais estão disponpiveis nesse [link](https://github.com/Skuldur/Classical-Piano-Composer)


# Output Sampling Methods in Music Generation  - EN

This project allows you to train a neural network to generate midi music files that make use of a single instrument

## Requirements

* **Is suggested** installing the Miniconda for python enviroment management
* Python 3.6.7
* Installing the packages using pip in requirements.txt file:
	```
	pip install -r requirements.txt
	```
* Install cuda 8.0.61
* Install cudnn 8.0 v7.1

## Training

To train the network you run **lstm.py**.

```
python lstm.py
```

The network will use every midi file in ./midi_songs to train the network. The midi files should only contain a single instrument to get the most out of the training.

**NOTE**: You can stop the process at any point in time and the weights from the latest completed epoch will be available for text generation purposes.

## Generating music

Once you have trained the network you can generate text using **predict.py**


```
python predict.py
```

You can run the prediction file right away using the **weights.hdf5** file
