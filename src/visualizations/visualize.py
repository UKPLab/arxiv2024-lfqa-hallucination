import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS


# function to read text file and generate word cloud
def generate_word_cloud(text):
    """Generate word cloud from text"""
    stopwords = ["question", "answer"] + list(STOPWORDS)
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110, stopwords=stopwords).generate(text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()


if __name__ == '__main__':
    filepath = 'src/data/'
    file_name = 'comp_reason.txt'
    with open(filepath+file_name, 'r') as f:
        text = f.read()
    generate_word_cloud(text)
