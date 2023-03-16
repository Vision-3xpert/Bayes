from flask import Flask, render_template, make_response, request
import matplotlib.pyplot as plt
import io
import numpy as np
from scipy.stats import beta
import base64
from matplotlib import cycler
colors = cycler('color',
                ['#EE6666', '#3388BB', '#9988DD',
                 '#EECC55', '#88BB44', '#FFBBBB'])
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # récupérez les valeurs des champs de formulaire
        n_trials_prior = int(request.form['n_trials_prior']) 
        n_successes_prior = int(request.form['n_successes_prior']) 
        n_trials_likelihood = int(request.form['n_trials_likelihood'])
        n_successes_likelihood = int(request.form['n_successes_likelihood'])
        n_failure_prior = n_trials_prior - n_successes_prior 

        # générer le graphique
        numSteps = 10000
        x = np.linspace(0, 1, numSteps)

        ## Lin_successeselihood function
        likelihood = []
        for i in x:
            event = i**n_successes_likelihood * (1 - i)**(n_trials_likelihood - n_successes_likelihood)
            likelihood.append(event)
        ## Just normalize lin_successeselihood to integrate to one (for purposes of plotting)
        likelihood = likelihood / sum(likelihood) * numSteps

        # Clear any previous plots
        plt.clf()
        plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
               axisbelow=True, grid=True, prop_cycle=colors)
        plt.rc('grid', color='w', linestyle='solid')
        plt.rc('xtick', direction='out', color='gray')
        plt.rc('ytick', direction='out', color='gray')
        plt.rc('patch', edgecolor='#E6E6E6')
        plt.rc('lines', linewidth=2)
        ##plot Likelihood
        plt.plot(x, likelihood)
        ##Plot Prior
        plt.plot(x, beta.pdf(x, n_successes_prior, n_failure_prior))
        ## Plot posterior
        plt.plot(x, beta.pdf(x, n_successes_likelihood + n_successes_prior, n_trials_likelihood - n_successes_likelihood + n_failure_prior), linestyle='dashed')
        ## Legend
        plt.legend([f"Likelihood - Binomial({n_trials_likelihood},{n_successes_likelihood})", f"Prior - Beta({n_successes_prior},{n_failure_prior})", f"Posterior - Beta"])

        # Stocker le graphique dans une variable BytesIO
        img = io.BytesIO()
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(img, format='png')
        img.seek(0)

        # Convertir le graphique en chaîne base64 pour l'afficher dans le template
        graph = base64.b64encode(img.getvalue()).decode()
        
        # Afficher le graphique et élever la fenêtre au premier plan
        fig.canvas.manager.show()
        plt.show()
        return render_template('index.html', graph=f"data:image/png;base64,{graph}")

    # renvoyer le formulaire vide
    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True, threaded = True)
