Testing
=======

Unit Tests
----------------------------------

Run unit tests with:

.. code-block:: bash

   ./scripts/test_unit.sh

End-to-end Tests
----------------------------------

Run end-to-end tests with:

.. code-block:: bash

   ./scripts/test_e2e.sh

Type Checking
----------------------------------

Perform type checking with:

.. code-block:: bash

   ./scripts/test_types.sh

Formatting
----------------------------------

Apply code formatting with:

.. code-block:: bash

   ./scripts/format.sh

Test End-to-end Scenarios
----------------------------------

The best way to experiment with different scenarios and parameters is to open the repository in VSCode and use the Python Test Explorer to run the ``e2e`` tests.

Alternatively, you can launch a specific test from the command line like so:

.. code-block:: bash

   python3 -m pytest tests/e2e/e2e_test.py::TestE2E::test_social_dr_success_d