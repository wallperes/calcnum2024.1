import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, gaussian_kde, binom
from scipy.integrate import quad
import math

# Dados das médias dos alunos
medias = [
    7.7, 6.8, 6.5, 6.2, 5.9, 5.5, 5.1, 5.1, 5.1, 4.6,
    4.3, 4.2, 4.1, 4.1, 3.9, 3.9, 3.7, 3.7, 3.3, 2.8,
    1.0, 1.0, 0.8, 0.7, 0.6, 0.6, 0.5, 0.2
]

# Estimativas dos parâmetros da distribuição bimodal
alpha_est = 0.2843
mu1_est = 0.6743
sigma1_est = 0.2485
mu2_est = 4.8173
sigma2_est = 1.2588

# Função de densidade de probabilidade da mistura de gaussianas
def bimodal_pdf(x, alpha, mu1, sigma1, mu2, sigma2):
    pdf1 = alpha * norm.pdf(x, mu1, sigma1)
    pdf2 = (1 - alpha) * norm.pdf(x, mu2, sigma2)
    return pdf1 + pdf2

# Função de distribuição acumulada (CDF) da mistura de gaussianas
def bimodal_cdf(x, alpha, mu1, sigma1, mu2, sigma2):
    cdf1 = alpha * norm.cdf(x, mu1, sigma1)
    cdf2 = (1 - alpha) * norm.cdf(x, mu2, sigma2)
    return cdf1 + cdf2

# Gerando os dados para a curva bimodal combinada
x = np.linspace(min(medias), max(medias), 1000)
bimodal_curve = bimodal_pdf(x, alpha_est, mu1_est, sigma1_est, mu2_est, sigma2_est)

# Estimando a função de densidade de probabilidade usando KDE com a regra de Scott ajustada
kde_scott = gaussian_kde(medias, bw_method='scott')
kde_scott.set_bandwidth(bw_method=kde_scott.factor * 0.5)

# Parâmetros estimados pelo KDE com a regra de Scott ajustada
bandwidth_scott = kde_scott.factor  # Largura de banda estimada

st.write(f"Parâmetros estimados pelo KDE com a regra de Scott ajustada:")
st.write(f"Largura de banda estimada: {bandwidth_scott:.4f}")

# Gerando os dados suavizados para plotagem
y_kde_scott = kde_scott(x)

# Plotando o histograma, a curva bimodal combinada e a curva KDE suavizada
fig, ax = plt.subplots(figsize=(10, 6))

# Histograma
ax.hist(medias, bins=10, density=True, edgecolor='black', alpha=0.7, label='Histograma')

# Curva bimodal combinada
ax.plot(x, bimodal_curve, color='red', label='Distribuição Bimodal')

# Curva KDE suavizada com a regra de Scott ajustada
ax.plot(x, y_kde_scott, color='blue', label='KDE Suavizado (Scott 0.5)')

ax.set_title('Distribuição das Médias dos Alunos')
ax.set_xlabel('Média')
ax.set_ylabel('Densidade de Probabilidade')
ax.legend()
ax.grid(True)

st.pyplot(fig)

# Widgets para coletar os limites do intervalo
lower_bound = st.number_input('Limite Inferior:', value=0.0)
upper_bound = st.number_input('Limite Superior:', value=10.0)

if st.button('Calcular Probabilidade'):
    # Calculando a probabilidade usando a CDF da mistura de gaussianas
    probabilidade_bimodal = bimodal_cdf(upper_bound, alpha_est, mu1_est, sigma1_est, mu2_est, sigma2_est) - \
                            bimodal_cdf(lower_bound, alpha_est, mu1_est, sigma1_est, mu2_est, sigma2_est)

    # Calculando a probabilidade usando a KDE
    probabilidade_kde, _ = quad(kde_scott, lower_bound, upper_bound)

    # Calculando a probabilidade ocorrida na turma
    num_alunos_intervalo = sum(lower_bound <= m <= upper_bound for m in medias)
    probabilidade_ocorrida = num_alunos_intervalo / len(medias)

    # Calculando intervalo de confiança para a probabilidade ocorrida
    n = len(medias)
    z = 1.96  # valor z para 95% de confiança
    p = probabilidade_ocorrida
    erro_maximo = 0.02  # 2%
    
    # Ajustando o intervalo de confiança para um erro máximo de 2%
    intervalo_confianca_inferior = max(0, p - erro_maximo)
    intervalo_confianca_superior = min(1, p + erro_maximo)
    
    # Verificando se as estimativas estão dentro do intervalo de confiança
    dentro_intervalo_bimodal = intervalo_confianca_inferior <= probabilidade_bimodal <= intervalo_confianca_superior
    dentro_intervalo_kde = intervalo_confianca_inferior <= probabilidade_kde <= intervalo_confianca_superior

    st.write(f"A probabilidade de um aluno ter média entre {lower_bound} e {upper_bound} é:")
    st.write(f"Usando a CDF da mistura de gaussianas: {probabilidade_bimodal:.4f} {'(Dentro do intervalo de confiança)' if dentro_intervalo_bimodal else '(Fora do intervalo de confiança)'}")
    st.write(f"Usando a KDE: {probabilidade_kde:.4f} {'(Dentro do intervalo de confiança)' if dentro_intervalo_kde else '(Fora do intervalo de confiança)'}")
    st.write(f"Probabilidade ocorrida na turma: {probabilidade_ocorrida:.4f}")
    st.write(f"Intervalo de confiança para a probabilidade ocorrida (95%): [{intervalo_confianca_inferior:.4f}, {intervalo_confianca_superior:.4f}]")
