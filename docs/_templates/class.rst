{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :inherited-members:
   :exclude-members: __weakref__, __dict__, __module__, __init__, get_metadata_routing, set_predict_request, set_params, get_params

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods
   {% endif %}
   {% endblock %}

.. bibliography::
   :filter: docname in docnames
   :style: unsrt