import pandas

if __name__ == '__main__':
    df = pandas.read_csv('logs/training_1.csv')
    ax = df.plot(x=0, title='Mean squared error loss')
    ax.annotate('Final model', xy=(12, df.iloc[11, 2]), xytext=(-40, 40), textcoords='offset points',
                         arrowprops=dict(arrowstyle="->", connectionstyle="angle3,angleA=0,angleB=-90"))
    ax.get_figure().savefig('writeup-img/loss.png')

