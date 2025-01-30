########
lamatrix
########

.. image:: https://github.com/christinahedges/lamatrix/actions/workflows/python-app.yml/badge.svg
    :target: https://github.com/christinahedges/lamatrix/actions/workflows/python-app.yml
    :alt: Test status

.. image:: https://badge.fury.io/py/lamatrix.svg
    :target: https://badge.fury.io/py/lamatrix
    :alt: PyPI version

.. image:: https://img.shields.io/badge/documentation-live-blue.svg
    :target: https://christinahedges.github.io/lamatrix/
    :alt: Documentation

.. <!-- intro content start -->

The ``lamatrix`` package is designed to help you build linear algebra models in Python to help you fit simple models to data. ``lamatrix`` will do all the fitting and shaping inside the package, so you can put in objects of any shape and fit ND models easily. 

.. <!-- intro content end -->

.. <!-- quickstart content start -->

Quickstart
==========

The easiest way to install ``lamatrix`` and all of its dependencies is to use the ``pip`` command.

To install ``lamatrix``, run the following command in a terminal window:

.. code-block:: console

  $ python -m pip install lamatrix --upgrade

The ``--upgrade`` flag is optional, but recommended if you already
have ``lamatrix`` installed and want to upgrade to the latest version.

Depending on the specific Python environment, you may need to replace ``python``
with the correct Python interpreter, e.g., ``python3``.

If you are trying to develop functionality to ``lamatrix`` you can install using github and poetry. Make sure you install the development dependencies

.. code-block:: console

  $ git clone https://github.com/PandoraMission/lamatrix
  $ cd lamatrix
  $ poetry install --with dev

You can then add functionality and run tests using

.. code-block:: console

  $ cd lamatrix
  $ make

If you install in this way you will need to use poetry to use the code. Check out the `poetry documentation`_ for more information on how to use poetry.

Now you have the package installed, you should check out the tutorials to see how to build a model with ``lamatrix`` and fit data. 

    .. _`poetry documentation`: https://python-poetry.org/docs/

.. <!-- quickstart content end -->

.. <!-- Contributing content start -->

Contributing
============

``lamatrix``  is an open-source package. Users are welcome to contribute and develop new features for ``lamatrix``, or add new documentation.

To work on ``lamatrix`` and add new functionality you can follow these steps.

0. Fork the project
-------------------

You should for the ``lamatrix`` github repo to your own account so that you can open pull requests against the main repository.

1. Installing the package
-------------------------

If you are trying to develop functionality to ``lamatrix`` you can install using github and poetry. Make sure you install the development dependencies

.. code-block:: console

  $ git clone https://github.com/PandoraMission/lamatrix
  $ cd lamatrix
  $ poetry install --with dev

This has now created a new poetry enviroment and installed the dependencies, development dependencies, and the package itself. 

2. Updating the package
-----------------------

You can now add any updates to the package you would like. The first step is to create and name a new branch.

.. code-block:: console
    $ git checkout -b BRANCHNAME

If you have not used your branch in a while, make sure you pull and merge any updates on the main branch.

.. code-block:: console
    $ git pull origin main

You will have to resolve any merge conflicts.

Once you have update the package, you should ensure you have done the following

* Updated the version number in the pyproject.toml file. If this is a bug fix, update the patch number. If it is adding new functionality but otherwise not changing the API update the minor number. If it is a change to the API entirely, update the major version number.
* Update the CHANGELOG in this readme file. 
* Ensure your functionality has updated documentation. This means both adding docstrings, and adding to the API documentation in the ``docs/`` directory. 
* Ensure your new functionality is covered by new tests. If you add any functionality, add tests in the ``tests/`` directory. 

3. Running tests
----------------

To run tests you can go into the root directory for the package and use the ``Makefile``.

.. code-block:: console
    $ make

This will run ``black``, ``isort``, ``flake8``, and ``pytest``. All of these should pass locally on your machine. Ensure the tests pass before moving to the next steps.

4. Building docs
----------------

The docs can be checked by running the following from within the ``docs/`` directory

.. code-block:: console
    $ make serve

This will compile and serve the docs at ``http://127.0.0.1:8001``. This will recompile all the notebooks in the ``docs/`` directory. You should be able to stop the serve action by using ``ctrl+c`` in the terminal window. If you accidentally close the terminal window and that port is blocked you can use

.. code-block:: console
    $ make stop-serve

to stop the docs on that port. 

Once you have checked the docs look good you can move onto the next step.

5. Opening a Pull Request
-------------------------

You can now open a pull request against the main branch on the main repository. Once the PR is reviewed and found to enhance the package, it will be merged in by an administrator. 

.. <!-- Contributing content end -->

.. <!-- Contact content start -->

Contact
=======

``lamatrix`` is an open source community package. The best way to notify a bug in the package is to `open an issue`_. Please include a self-contained example that fully demonstrates your problem or question.

  .. _`open an issue`: https://github.com/christinahedges/lamatrix/issues/new

.. <!-- Contact content end -->

.. <!-- Changelog content start -->

Changelog:
==========

  - Initial v1.0.0 release of `lamatrix`.
  
.. <!-- Changelog content end -->