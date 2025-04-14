# heat_transfer2d3d

Interface para identificar a transição da transferência de calor de 3d para 2d.

Feita usando uma rede neural para identificação das temperaturas de pico e posteriormente verificação das posições em X onde a temperatura não muda (ou muda pouco).

Criar ambiente virual em python (p/ Windows):

```python -m venv env```

Ativar o ambiente virtual (p/ Windows):

```.\env\Scripts\activate```

Instalação das dependências:

```pip install -r requirements.txt```

Para rodar o software:

```python main.py```