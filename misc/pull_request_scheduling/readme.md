### Some tips for reviewing
*  The code should run, and do what it claims to. If there are no tests included with the merge request,
   ask the author to write some, or write some of your own. Please try to use <a href="http://docs.pytest.org">PyTest</a>,
   so that with time we can automate the testing of the system.
*  The program should be as clear as possible. If there are any parts you find mysterious, ask.
*  Comments should help explain how the piece of a program works, without being a pseudocode version of the program. Complex,
   or otherwise hard to understand, parts of the program merit more comments, and conversely, simple or routine
   parts do not require many comments. Justifications for certain design decisions are also good candidates for
   comments.
*  Style-wise, <a href="https://www.python.org/dev/peps/pep-0008/">pep8</a> is pretty good and standard,
   and should mostly be followed.


### Steps for scheduling a merge request:

Going through the following steps is optional. If you 
do not, then I (that is, Ilia) will do them for you.
But please, do everything on the list if you do decide to do some of it.

The following steps are taken to schedule and register a MR review (there is a lot of intentional redundancy):
<ul>
    <li> Edit the <tt>prs.json</tt> file.
         Include a new entry in the "pull_requests" list (copy/paste is suggested). The entry should be a dictionary with fields
         <ul> 
             <li> "id": Merge request number </li>
             <li> "title": Merge request title </li>
             <li> "author": Merge request author </li>
             <li> "reviewers": [] or null
        </ul>
        The last point is important, because it is used to check whether or not the request has already been assigned. <br>
        Just in case you have not worked with it before, json is very similar to a
         Python dictionary, but you cannot leave
         trailing commas, have to use double quotes "", and have to use <tt>null</tt> instead of <tt>None</tt>, among other minor
         differences.
  </li>
  <li> Run the <tt>queue.py</tt> script. The script will schedule the requests that 
       do not have assigned reviewers in the json file.
       The script is wise and all-knowing, and its decisions can never be questioned (just kidding)
  </li>
  <li> Make announcements:
         <ul>
             <li> <a href="https://trello.com/b/jbik36ZO/wolf-project">Trello</a>:
                          Add a ticket to the Pull Requests board. Assign the Author, and both Reviewers to the ticket,
                          and include a link to the Pull Request page on gitlab. </li>
             <li> Gitlab: tag both reviewers in a comment on the merge request to let them know they have been assigned to it. </li>
             <li> Gitlab (again): Add the new requests to the table in the <tt>README.md</tt> file on the 
                  <a href="http://116.66.187.35:4502/gitlab/its/sow45_code">gitlab root directory</a>. </li>
             <li> Slack (Optional): Make an announcement in the <tt>pull-requests</tt> channel.</li>
         </ul>
  </li>
</ul>
Whew!
