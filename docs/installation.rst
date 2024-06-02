.. highlight:: shell

============
Installation
============


Stable release
--------------

To install `vac_aligner`, run this command in your terminal:

.. code-block:: console

    $ pip install vac_aligner['full']

This is the preferred method to install vac_aligner, as it will always install the most recent stable release.

The 2nd part of our pipeline requires  `nemo-toolkit['asr']` which can make the library very heavy,
especially, for GPU compatible torch version. Thus, you can skip this **extra** installations.

.. code-block:: bash

   pip install vac_aligner

Now you can use VAD or the Matching Part (say you have your own ASR model or predictions) while skipping the `torch` and `nemo` installations!


If you don't have `pip`_ installed, this `Python installation guide`_ can guide
you through the process.

.. _pip: https://pip.pypa.io
.. _Python installation guide: http://docs.python-guide.org/en/latest/starting/installation/


From sources
------------

The sources for vac_aligner can be downloaded from the `Github repo`_.

You can either clone the public repository:

.. code-block:: console

    $ git clone git://github.com/Ara-Yeroyan/vac_aligner

Or download the `tarball`_:

.. code-block:: console

    $ curl -OJL https://github.com/Ara-Yeroyan/vac_aligner/tarball/master

Once you have a copy of the source, you can install it with:

.. code-block:: console

    $ python setup.py install


.. _Github repo: https://github.com/Ara-Yeroyan/vac_aligner
.. _tarball: https://github.com/Ara-Yeroyan/vac_aligner/tarball/master
