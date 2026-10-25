# Experimento de convergência

As novas células de análise do notebook traduzido usam `../convergencia.py` para
estender o experimento de 200 a 2.000 atualizações. Os resultados da Parte 3
original permanecem em `../resultados_parte3/`.

Execute as células depois de carregar os conjuntos de treinamento ou, a partir
da raiz do repositório, use:

```bash
python trab2/convergencia.py --max-iter 2000
```

A linha de comando reproduz a amostragem original: semente 42, primeiro a
subamostra Partial e depois Scarce. O notebook usa os arrays já carregados;
hashes diferentes geram experimentos separados em vez de misturar resultados.

Cada subpasta identificada pelo hash do experimento contém:

- `config.json`: dados, implementação de treinamento, versões, sementes, folds,
  taxa de aprendizado, grade, otimizador e convenções das métricas.
- `request.json`: orçamento solicitado e frequência de checkpoint.
- `status.json`: `complete` somente quando todos os folds terminaram o orçamento;
  `incomplete` inclui execuções em andamento ou interrompidas.
- `splits.csv`: regime, índice da amostra dentro do regime e fold em que ela é
  validação. Cada amostra aparece uma vez; os demais folds a usam no treino.
- `<regime>/fold_<n>/iterations.csv`: métricas de todos os candidatos por iteração.
- `<regime>/fold_<n>/checkpoint.pt`: pesos, bias, estado do SGD e última iteração
  confirmada. Os candidatos estão na ordem registrada em `config.json`.
- `iterations.csv`: consolidação dos nove folds, criada ao completar a execução.
- `cv_by_iteration.csv`: médias e desvios amostrais entre folds por configuração
  e iteração; somente iterações com todos os folds são agregadas.
- `selected_by_budget.csv`: seleção por acurácia de CV nos orçamentos fixos 200,
  500, 1.000 e 2.000 (desempate pelo menor λ).
- `convergence_diagnostics.csv`: variação do objetivo e métricas da janela final
  de 100 atualizações, por fold. São indicadores, não certificados de convergência.
- `comparison_200.csv`: comparação com os resultados originais na mesma iteração,
  gerada pela célula de análise do notebook.
- `figures/`: curvas de objetivo de treino, CE de treino/validação e acurácia de
  validação, uma figura por regime, com colunas para L1 e L2. As perdas usam
  escala logarítmica; os valores registrados no CSV não são transformados.

## Convenções

A iteração 0 representa a inicialização. Todas as métricas na linha `t` são
avaliadas **depois de t atualizações**. `train_ce` e `val_ce` são entropias
cruzadas médias, sem penalidade. `train_objective` soma a CE de treino com
`regularization_term = lambda * penalty`. `penalty` é a soma de valores absolutos
(L1) ou quadrados (L2) dos pesos; o bias não é penalizado. Para o baseline,
`penalty` e `regularization_term` são zero.

Acurácias e frações de pesos são proporções, não porcentagens.
`near_zero_fraction` usa `abs(weight) < 1e-3`. `gradient_norm` inclui pesos e
bias no estado atual e usa o subgradiente do PyTorch para L1. `update_norm`
mede a atualização que chegou ao estado atual; `relative_update_norm` a divide
pela norma dos parâmetros anteriores. Ambas valem zero na inicialização.
`elapsed_seconds` é o tempo acumulado por fold, compartilhado pelos candidatos,
incluindo avaliação e persistência, mas sem o tempo entre sessões interrompidas.

Cada modelo recebe o gradiente do seu próprio objetivo: a soma dos objetivos
independentes permite avaliar todos em conjunto sem dividir o gradiente pelo
número de candidatos. Todos começam com o mesmo `nn.Linear` no fold, usando
semente `42 + fold - 1`. Os produtos matriciais conjuntos podem mudar ligeiramente
o arredondamento em float32 em relação ao treinamento individual.

O baseline é treinado uma vez por fold e aparece como `regularizer=Baseline` no
CSV bruto. A agregação o reutiliza como λ=0 para L1 e L2; isso não cria folds
adicionais nem observações independentes.

## Cache e retomada

CSV e checkpoint são substituídos atomicamente, a cada 50 iterações e ao terminar.
O CSV é gravado primeiro. Após uma interrupção entre gravações, registros além
do último checkpoint são descartados e recalculados. Até 50 atualizações não confirmadas podem ser refeitas se a interrupção ocorrer
durante a gravação de um checkpoint. Um fold concluído é lido do cache sem treinamento.

Aumentar `--max-iter` ou `conv_max_iter` continua a mesma execução. Dados,
hiperparâmetros, versões ou implementação de treinamento diferentes geram outro
hash. Alterar somente funções de análise não invalida o treinamento armazenado.
Não execute dois escritores simultâneos na mesma pasta de experimento.

Os testes de equivalência dos gradientes, retomada e consistência dos registros
podem ser executados com `python -m unittest discover -s trab2 -p test_convergencia.py -v`
a partir da raiz do repositório.

Nenhum conjunto de teste é utilizado. O desvio entre folds não é um intervalo
de confiança. Os modelos desta análise são modelos de folds; os modelos finais
treinados com todos os dados da Parte 3 não são substituídos automaticamente.

## Conferência da execução incluída

O experimento `766e3d63eefab0a3` concluiu os nove folds até 2.000 atualizações,
com 198.099 registros brutos (incluindo a inicialização), sem duplicatas ou
valores não finitos. A identidade `train_objective = train_ce + regularization_term`
foi conferida com tolerância numérica de float32.

Em 200 atualizações, 35 das 36 configurações coincidem com a acurácia média de
validação original até tolerância absoluta de `1e-6`. A exceção é Full/L1/λ=0,1,
com diferença de +0,00035 em fração (+0,035 ponto percentual). Essa diferença
entre execuções está registrada em `comparison_200.csv`; ela não é um efeito
do aumento do orçamento, pois compara a mesma iteração. O treinamento conjunto
pode alterar o arredondamento dos produtos em float32, e o SGD com L1 e passo
fixo pode ser sensível a essas diferenças. A referência original foi preservada.
