{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :show-inheritance:
   :inherited-members:
   :exclude-members: __weakref__, __dict__, __module__, __init__, get_metadata_routing, set_predict_request, set_params, get_params

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods

   .. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

.. bibliography::
   :filter: docname in docnames
   :style: unsrt