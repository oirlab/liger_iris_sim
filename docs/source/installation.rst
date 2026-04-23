Installation
============

Python Environment
------------------

It is highly recommended users create a new Python environment with either `venv <https://docs.python.org/3/library/venv.html>`_, `Anaconda <https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html>`_, or `uv <https://docs.astral.sh/uv/pip/environments/>`_. The pipeline supports Python versions 3.12 or newer.

To create a **uv** environment specifically for the latest stable release of ``liger_iris_sim`` (in this example, called **liger_iris**)::

    uv venv liger_iris --python 3.13

This will create a fresh Python 3.13 environment.

Once you have created your **uv** environment, activate it::

    source liger_iris/bin/activate


Install from GitHub
-------------------

1. Clone the repository:

.. code-block:: bash

    git clone https://github.com/astrobc1/liger_iris_sim.git

2. Install live version with pip:

.. code-block:: bash

   cd liger_iris_sim
   uv pip install -e .

The ``-e`` (editable) flag uses the specified package directory as the "live" source code, and does not copy it to the usual site-packages directory.

Configure Resources Directory
-----------------------------

A local directory is used for extra data resources like filter curves, PSFs, and model spectra.
It can be specified with the environment variable

**LIGER_IRIS_DRP_RESOURCE_DIR**

The default location is the result of:

.. code-block:: python

  from astropy.utils.data import _get_download_cache_loc
  import os
  resource_dir = os.path.join(_get_download_cache_loc(), 'LIGER_IRIS_DRP_RESOURCES')

This will usually be ``~/.astropy/cache/download/url/LIGER_IRIS_DRP_RESOURCES/``.

Required! - Download Resources
++++++++++++++++++++++++++++++

Download the resources with:

.. code-block:: python

  from liger_iris_drp_resources import download
  download(
    filter_trans=True,
    liger_psfs=True,
    iris_psfs=False # NOT YET AVAILABLE
    model_spectra=True
    skip_if_exists=True
  )