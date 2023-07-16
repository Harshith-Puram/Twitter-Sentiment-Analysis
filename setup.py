import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"

REPO_NAME = "Twitter-Sentiment-Analysis"
AUTHOR_USER_NAME = "Harshith-Puram"
SECOND_AUTHOR_USER_NAME = "Druvika-N"
AUTHOR_EMAIL = "harshithppuram@gmail.com"
SECOND_AUTHOR_EMAIL = "druvikan@gmail.com"
SRC_REPO = "SentimentAnalysis"



setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_USER_NAME,
    author_email=AUTHOR_EMAIL,
    description="A small python package for Sentiment Analysis app",
    long_description=long_description,
    long_description_content="text/markdown",
    url=f"https://github.com/Harshith-Puram/Twitter-Sentiment-Analysis",
    project_urls={
        "Bug Tracker": f"https://github.com/Harshith-Puram/Twitter-Sentiment-Analysis/issues",
    },
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src")
)