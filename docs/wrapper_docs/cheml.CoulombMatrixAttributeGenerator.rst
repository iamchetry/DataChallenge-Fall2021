.. _CoulombMatrixAttributeGenerator:

CoulombMatrixAttributeGenerator
================================

:task:
    | Represent

:subtask:
    | inorganic descriptors

:host:
    | chemml

:function:
    | CoulombMatrixAttributeGenerator

:input tokens (receivers):
    | ``entries`` : list of entries from CrystalStructureEntry class.
    |   types: ("<type 'list'>",)

:output tokens (senders):
    | ``df`` : pandas dataframe
    |   types: ("<class 'pandas.core.frame.DataFrame'>",)


:required packages:
    | ChemML, 0.4.1
    | pandas, 0.20.3

:config file view:
    | ``##``
    |   ``<< host = chemml    << function = CoulombMatrixAttributeGenerator``
    |   ``<< n_eigenvalues = 30``
    |   ``>> id entries``
    |   ``>> id df``
    |
    .. note:: The documentation page for function parameters: 
