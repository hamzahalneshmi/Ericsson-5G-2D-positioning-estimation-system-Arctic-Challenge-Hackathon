from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from scipy.optimize import least_squares
import plotly.graph_objs as go

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    bs_file = request.files['bs_data']
    toa_file = request.files['toa_data']

    bs_positions = np.genfromtxt(bs_file, delimiter=',')
    toa_data = np.genfromtxt(toa_file, delimiter=',')

    estimated_positions = perform_position_estimation(bs_positions, toa_data)

    plot_div = plot_results(bs_positions, estimated_positions)

    return render_template('result.html', plot_div=plot_div)

@app.route('/run_example')
def run_example():
    bs_positions = np.genfromtxt('positioning_data_BS_positions.csv', delimiter=',')
    toa_data = np.genfromtxt('positioning_data_ToA_measurements.csv', delimiter=',')

    estimated_positions = perform_position_estimation(bs_positions, toa_data)

    plot_div = plot_results(bs_positions, estimated_positions)

    return render_template('result.html', plot_div=plot_div)

def perform_position_estimation(bs_positions, toa_data):
    N_BS = bs_positions.shape[1]
    T = toa_data.shape[0]

    def residuals(params, anchors, toa_measurements):
        x, y = params
        d = np.sqrt((anchors[0, :] - x)**2 + (anchors[1, :] - y)**2)
        return d - toa_measurements

    estimated_positions = np.zeros((T, 2))

    for t in range(T):
        reference_bs = bs_positions[:, 0]
        toa_diff = toa_data[t, :] - toa_data[t, 0]
        initial_position = reference_bs
        result = least_squares(residuals, initial_position, args=(bs_positions, toa_diff))
        estimated_positions[t, :] = result.x

    return estimated_positions

def plot_results(bs_positions, estimated_positions):
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=bs_positions[0, :],
        y=bs_positions[1, :],
        mode='markers',
        name='Base Stations',
        marker=dict(color='blue')
    ))

    fig.add_trace(go.Scatter(
        x=estimated_positions[:, 0],
        y=estimated_positions[:, 1],
        mode='markers',
        name='Estimated Positions',
        marker=dict(color='red', symbol='x')
    ))

    fig.update_layout(
        title='2D Position Estimation',
        xaxis=dict(title='X-coordinate'),
        yaxis=dict(title='Y-coordinate'),
        legend=dict(x=0, y=1),
        grid=dict(rows=1, columns=1, pattern='independent'),
    )

    plot_div = fig.to_html(full_html=False)

    return plot_div

if __name__ == '__main__':
    app.run(debug=True)
