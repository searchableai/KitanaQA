# Contributing to KitanaQA
Everyone is welcome to contribute, and we value everybody's contribution. Code is thus not the only way to help the community. 
Answering questions, helping others, reaching out and improving the documentations are immensely valuable to the community.

It also helps us if you spread the word. How? 

- Reference the library from blog posts on the awesome projects it made possible
- Shout out on Twitter every time it has helped you
- Or simply star the repo to say "thank you".

## You can contribute in so many ways!

There are 4 ways you can contribute to KitanaQA:

- Fixing outstanding issues with the existing code
- Implementing new models
- Contributing to the examples or to the documentation
- Submitting issues related to bugs or desired new features
- All are equally valuable to the community

# Submitting a new issue or feature request
Do your best to follow these guidelines when submitting an issue or a feature request. It will make it easier for us to come back to you quickly and with good feedback.

# Found a bug?

Our goal is to keep KitanaQA robust and reliable thanks to the users who notify us of the problems they encounter. So thank you for reporting an issue.

First, we would really appreciate it if you could make sure the bug was not already reported (use the search bar on Github under Issues).

Did you not find it? Click [here](https://github.com/searchableai/KitanaQA/issues) to go there direactly. 

In order that we can address your issue or pull request quickly, please follow these steps:

- Include your *OS type* and *version*, the versions of Python, PyTorch and Tensorflow when applicable
- What the exact issue is that you are facing, along with an error message, if any
- What you would expect to happen instead
- If you have tried any other alternative approaches or methods
- A short, self-contained, code snippet that allows us to reproduce the bug
- If you don't have a code snippet, please go step by step about how someone can recreate that issue

# Want to implement a new model?
Awesome, that's great! To get started, please provide the following information about,

- Short description of the model and link to the paper
- Link to the implementation if it is open-source
- Link to the model weights if they are available
- If you are willing to contribute the model yourself, let us know so we can best guide you.

## OR, do you want a new feature (that is not a model)?
A world-class feature request addresses the following points:

### Motivation
- Is it related to a problem/frustration with the library? If so, please explain why. Providing a code snippet that demonstrates the problem is best
- Is it related to something you would need for a project? We'd love to hear about it!
- Is it something you worked on and think could benefit the community? Awesome! Tell us what problem it solved for you
- Write a full paragraph describing the feature
- Provide a code snippet that demonstrates its future use
- In case this is related to a paper, please attach a lin;
- Attach any additional information (drawings, screenshots, etc.) you think may help
- If your issue is well written we're already 80% of the way there by the time you post it

## Start contributing! (Pull Requests)
Before writing code, we strongly advise you to search through the exising PRs or issues to make sure that nobody is already working on the same thing. If you are unsure, it is always a good idea to open an issue to get some feedback.

You will need basic git proficiency to be able to contribute to transformers. git is not the easiest tool to use but it has the greatest manual. Type git --help in a shell and enjoy. If you prefer books, Pro Git is a very good reference.

If this is your first time contributing to an open source project, check out [First Contributions](https://github.com/firstcontributions/first-contributions)!

### Follow these steps to start contributing

Please follow these steps to start contributing,

1. Create an account on GitHub if you do not already have one
2. Fork the [project repository](https://github.com/searchableai/KitanaQA) - click on the ‘Fork’ button near the top of the page. This creates a copy of the code under your account on the GitHub user account. For more details on how to fork a repository see this [guide](https://help.github.com/articles/fork-a-repo/).
3. Clone your fork of the KitanaQA repo from your GitHub account to your local disk:
```bash
$ git clone https://github.com/${YourLogin}/KitanaQA.git # add --depth 1 if your connection is slow
$ cd KitanaQA
```
4. Add the `upstream` remote. This saves a reference to the main KitanaQA repository, which you can use to keep your repository synchronized with the latest changes:
```bash
git remote add upstream https://github.com/searchableai/KitanaQA.git
```

You should now have a working installation of KitanaQA, and your git repository properly configured. The next steps now describe the process of modifying code and submitting a PR:
5. Synchronize your master branch with the upstream master branch:
```bash
$ git checkout main
$ git pull upstream main
```
6. Create a feature branch to hold your development changes:
```bash
$ git checkout -b my_feature
```
and start making changes. Always use a feature branch. It’s good practice to never work on the `main` branch!
7. Develop the feature on your feature branch on your computer, using Git to do the version control. When you’re done editing, add changed files using `git add` and then `git commit`:
```bash
$ git add modified_files
$ git commit
```
to record your changes in Git, then push the changes to your GitHub account with:
```bash
$ git push -u origin my_feature
```

Follow [these](https://docs.github.com/en/free-pro-team@latest/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request-from-a-fork) instructions to create a pull request from your fork. This will send an email to the committers. You may want to consider sending an email to the mailing list for more visibility.

It is often helpful to keep your local feature branch synchronized with the latest changes of the main scikit-learn repository,
```bash
$ git fetch upstream
$ git merge upstream/master
```

And you're done! If you have any questions, feel free to ask the maintainers for guidance. 
This guide was heavily inspired by the awesome Hugging Face Transformers guide to contributing.

Happy coding!
