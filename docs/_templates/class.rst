{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
   :members:
   :exclude-members: __weakref__, __dict__, __module__, __init__, set_predict_request, set_inverse_transform_request, set_score_request 

   {% block methods %}

   {% if methods %}
   .. rubric:: Methods
   {% endif %}
   {% endblock %}

.. bibliography::
   :keyprefix: {{ objname | lower }}-
   :style: unsrt