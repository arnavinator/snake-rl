import matplotlib.pyplot as plt
# from IPython import display

plt.ion()  # interactive mode on

def plot(scores, mean_scores, show_final=False):
    # display.clear_output(wait=True)
    # display.display(plt.gcf())  # get current fig
    plt.clf()                   # clear fig
    plt.title('Score vs # Training Games')
    plt.xlabel('# of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    # annotate the most recent scores
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(.1)

    if show_final:
        plt.ioff()
        plt.show(block=True)
