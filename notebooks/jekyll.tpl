{% extends 'markdown.tpl' %}

{%- block header -%}
---
layout: page
title: "Tensor Considered Harmful"
excerpt: "Named tensors for better deep learning code."
---
{%- endblock header -%}


{% block input %}
{{ '{% highlight python %}' }}
{{ cell.source }}
{{ '{% endhighlight %}' }}
{% endblock input %}

{% block data_svg %}
![svg]({{ output.metadata.filenames['image/svg+xml'] | path2support }})
{% endblock data_svg %}

{% block data_text %}
{:.output_data_text}
```
{{ output.data['text/plain'] }}
```
{% endblock data_text %}

{% block data_png %}
![png]({{ output.metadata.filenames['image/png'] | path2support }})
{% endblock data_png %}

{% block data_jpg %}
![jpeg]({{ output.metadata.filenames['image/jpeg'] | path2support }})
{% endblock data_jpg %}

{% block markdowncell scoped %}
{{ cell.source | wrap_text(80) }}
{% endblock markdowncell %}

{% block headingcell scoped %}
{{ '#' * cell.level }} {{ cell.source | replace('\n', ' ') }}
{% endblock headingcell %}
