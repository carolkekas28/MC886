# Experimentos da Parte 3

Gerados pelas celulas da Parte 3 do notebook traduzido. Nenhum conjunto de teste e usado nesta etapa.

- `cv_folds.csv`: acuracias de treino e validacao, tamanhos das particoes e identificador do fold, por regime, penalizacao e lambda.
- `cv_results.csv`: media e desvio padrao amostral de validacao (`ddof=1`), media de treino e diferenca treino-validacao.
- `selected_cv.csv`: candidato com maior media de validacao para cada regime e penalizacao. Empates exatos usam o menor lambda.
- `full_weight_stats.csv`: fracao de pesos com valor absoluto menor que `1e-3`, fracao de zeros exatos, media dos valores absolutos e norma L2 dos pesos, para todos os candidatos Full.
- `modelos_parte3.pt`: configuracao, lambdas selecionados e `state_dict` dos modelos finais selecionados e de todos os candidatos Full. Os parametros sao salvos em CPU.

Os graficos ficam em `../imagens_respostas/parte3_*.png` e nas saidas do notebook. Todos os valores de acuracia nos CSVs sao fracoes, nao porcentagens. A grade inclui zero para comparar com a ausencia de regularizacao.

Para recuperar um modelo em uma nova sessao, execute primeiro as definicoes de `LinearModel` do notebook e use:

```python
bundle = torch.load("resultados_parte3/modelos_parte3.pt", map_location="cpu", weights_only=True)
model = LinearModel()
model.load_state_dict(bundle["selected_state_dicts"]["Full"]["L1"])
model.eval()
```

Os resultados correspondem a 200 iteracoes de SGD completo, taxa de aprendizado 0.1 e tres folds estratificados. A comparacao e para esse orcamento fixo, sem afirmar convergencia de todos os candidatos. A fracao proxima de zero nao deve ser confundida com zeros exatos: L1 com SGD de passo fixo usa subgradientes, sem limiarizacao proximal.
