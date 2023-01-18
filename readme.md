### Merge request review

<table>
<tr>
    <th> id </th>
    <th> Author </th>
    <th> Title </th>
    <th> Reviewers </th>
</tr>
<tr>
    <td> 40 </td>
    <td> Xiaoyu </td>
    <td> agent info sharing </td>
    <td> Zhicheng, Parth </td>
</tr>
<tr>
    <td> 12 </td>
    <td> Parth </td>
    <td> dtse, dreamer experiments </td>
    <td> Ilia, Xiaoyu </td>
</tr>
<tr>
    <td> 38 </td>
    <td> Parth </td>
    <td> mb-exp: Model-based experiments</td>
    <td> Xiaoyu, Zhicheng </td>
</tr>
<tr>
    <td> 37 </td>
    <td> Parth </td>
    <td> Yellow phase added in benchmarks; pytest for benchmarks added </td>
    <td> Zhicheng, Ilia </td>
</tr>
<tr>
    <td> 32 </td>
    <td> Zhicheng </td>
    <td> Fix offset policy </td>
    <td> Xiaoyu, Parth </td>
</tr>
<tr>
    <td> 21 </td>
    <td> Zhicheng </td>
    <td> Max pressure policy </td>
    <td> Ilia, Xiaoyu </td>
</tr>
<tr>
    <td> 20 </td>
    <td> Xiaoyu </td>
    <td> Custom metric module </td>
    <td> Parth, Ilia </td>
</tr>
</table>

<a href="http://116.66.187.35:4502/gitlab/its/sow45_code/tree/master/misc/pull_request_scheduling">Steps for scheduling merge request reviews, and some tips for reviewing</a>

### Documentation

To build the documentation, under docs, type:

make html

Then open index.html under docs/build/html with your navigator.

### Troubleshooting

Extension error:
Could not import extension recommonmark (exception: No module named 'recommonmark')
Makefile:20: recipe for target 'html' failed

-> pip install recommonmark
