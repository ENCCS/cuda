Heterogeneous programming with CUDA/HIP
=======================================


Intro



.. prereq::

   prerequisites


.. toctree::
   :maxdepth: 1
   :caption: Table of contents

   1.01_GPUIntroduction
   2.01_DeviceQuery
   2.02_HelloGPU
   2.03_VectorAdd
   2.04_HeatEquation
   3.01_ParallelReduction
   3.02_TaskParallelism
   4.01_FromCUDAToHIP


.. toctree::
   :maxdepth: 1
   :caption: Reference

   quick-reference
   guide



.. _learner-personas:

Who is the course for?
----------------------

This course is for students, researchers, engineers and programmers who would like to learn GPU programming with CUDA.
Some previous experience with C/C++ is required, no prior knowledge of CUDA is needed.
  
Tentative schedule
------------------

.. list-table::
   :widths: 25 70
   :header-rows: 1

   * - Day 1
     - Thursday, October 7, 2021
   * - 9:00 -  9:10
     - Welcome and introduction to the training course
   * - 9:10 -  9:30
     - Introduction to GPUs
   * - 9:30 -  10:10
     - Using CUDA
   * - 10:10 - 10:20
     - Break
   * - 10:20 - 10:50
     - Adding two vectors on the GPU
   * - 10:50 - 11:10
     - Break-out rooms
   * - 11:10 - 11:20
     - Break
   * - 11:20 - 12:30
     - Solving heat equation with CUDA
   * - 12:30 - 12:50
     - Break-out rooms
   * - 12:50 - 13:00
     - Wrap-up

.. list-table::
   :widths: 25 70
   :header-rows: 1

   * - Day 2
     - Friday, October 8, 2021
   * - 9:00 -  9:10
     - Follow-ups from day 1
   * - 9:10 -  10:10
     - Optimizing the CUDA kernel
   * - 10:10 - 10:30
     - Break-out rooms
   * - 10:30 - 10:50
     - Optimizing the CUDA kernel (cont.)
   * - 10:50 - 11:00
     - Break
   * - 11:00 - 11:50
     - Exploring task-based parallelism
   * - 11:50 - 12:10
     - Break-out rooms
   * - 12:10 - 12:50
     - Exploring task-based parallelism (cont.)
   * - 12:50 - 13:00
     - Wrap-up

About the course
----------------

These course materials are developed for those who wants to leark GPU programming with CUDA from the beginning.
The course consists of lectures, type-along and hands-on sessions.

During the first day, we will cover the architecture of the GPU accelerators, basic usage of CUDA, and how to control data movement between CPUs and GPUs.
The second day focuses on more advanced topics, such as how to optimize computational kernels for efficient execution on GPU hardware and how to explore the task-based parallelism using streams and events.
We will also briefly go through profiling tools that can help one to identify the computational bottleneck of the applications.

After the course the participants should have the basic skills needed for using CUDA in new or existing applications.

The participants are assumed to have knowledge of C programming language.
Since participants will be using HPC clusters to run the examples, fluent operation in a Linux/Unix environment is assumed.


See also
--------



Credits
-------

The lesson file structure and browsing layout is inspired by and derived from
`work <https://github.com/coderefinery/sphinx-lesson>`_ by `CodeRefinery
<https://coderefinery.org/>`_ licensed under the `MIT license
<http://opensource.org/licenses/mit-license.html>`_. We have copied and adapted
most of their license text.

Instructional Material
^^^^^^^^^^^^^^^^^^^^^^

This instructional material is made available under the
`Creative Commons Attribution license (CC-BY-4.0) <https://creativecommons.org/licenses/by/4.0/>`_.
The following is a human-readable summary of (and not a substitute for) the
`full legal text of the CC-BY-4.0 license
<https://creativecommons.org/licenses/by/4.0/legalcode>`_.
You are free to:

- **share** - copy and redistribute the material in any medium or format
- **adapt** - remix, transform, and build upon the material for any purpose,
  even commercially.

The licensor cannot revoke these freedoms as long as you follow these license terms:

- **Attribution** - You must give appropriate credit (mentioning that your work
  is derived from work that is Copyright (c) Artem Zhmurov and individual contributors and, where practical, linking
  to `<https://enccs.github.io/CUDA>`_), provide a `link to the license
  <https://creativecommons.org/licenses/by/4.0/>`_, and indicate if changes were
  made. You may do so in any reasonable manner, but not in any way that suggests
  the licensor endorses you or your use.
- **No additional restrictions** - You may not apply legal terms or
  technological measures that legally restrict others from doing anything the
  license permits.

With the understanding that:

- You do not have to comply with the license for elements of the material in
  the public domain or where your use is permitted by an applicable exception
  or limitation.
- No warranties are given. The license may not give you all of the permissions
  necessary for your intended use. For example, other rights such as
  publicity, privacy, or moral rights may limit how you use the material.



Software
^^^^^^^^

Except where otherwise noted, the example programs and other software provided
with this repository are made available under the `OSI <http://opensource.org/>`_-approved
`MIT license <https://opensource.org/licenses/mit-license.html>`_.

