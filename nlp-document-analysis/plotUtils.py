import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pylab import rcParams
rcParams['figure.figsize'] = 20, 10


def plot_bar(dfx, dfy, labx, laby, filename):
	# plt.figure(figsize=(20,10))
	fig = plt.figure()
	plt.bar(dfx, dfy)
	plt.xticks(rotation=90)
	# plt.xlabel(labx)
	# plt.ylabel(laby)
	# plt.xticks(dfx, dfx)
	# plt.yticks(dfy)
	# plt.title('Most Used 20 Words')
	plt.tight_layout()
	plt.savefig(filename+'.png')

def plot_word_cloud(text, filename):
	# Generate a word cloud image
	wordcloud = WordCloud().generate(text)

	# Display the generated image:
	# the matplotlib way:
	plt.imshow(wordcloud, interpolation='bilinear')
	plt.axis("off")

	# lower max_font_size
	wordcloud = WordCloud(max_font_size=2000, width=1000, height=860, margin=2, background_color="white").generate(text)
	plt.figure()
	plt.imshow(wordcloud, interpolation="bilinear")
	plt.axis("off")
	plt.savefig(filename+'.png')