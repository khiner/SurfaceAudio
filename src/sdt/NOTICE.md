# SDT attribution

The SDT contact, body, and controller algorithms in this directory are adapted from commit `0509de418e7bebc8b37866b3b4458e0acc8cf1f4` of https://github.com/SkAT-VG/SDT.
The adaptations were made September 8, 2026 and include C++ data structures, shared C++/Metal scalar stepping, an explicitly named unilateral impact extension, an analytic solution of the source contact-energy inequality, and compensated GPU state-coordinate evaluation.
The following upstream license notice and author list are retained verbatim.
The complete GNU GPL version 3 is available in the repository root `LICENSE`.

```text
-------------------------------------------------------------------------------
Sound Design Toolkit (SDT)
--
https://github.com/SkAT-VG/SDT
-------------------------------------------------------------------------------

Copyright (C) 2001 - 2024 with the authors (see AUTHORS.txt)


This file is part of the Sound Design Toolkit (SDT).

The SDT is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The SDT is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the SDT.  If not, see <https://www.gnu.org/licenses/>.

The official SDT distribution contains copy of the GNU General Public License
in the file COPYING.txt
```

```text
-------------------------------------------------------------------------------
Sound Design Toolkit (SDT)
--
https://github.com/SkAT-VG/SDT
-------------------------------------------------------------------------------

Authors and contributors (either programmers or designers) in alphabetical
order:

	Federico Avanzini (federico.avanzini@di.unimi.it)
	Stefano Baldan (singintime@gmail.com)
	Nicola Bernardini
	Gianpaolo Borin
	Carlo Drioli (carlo.drioli@uniud.it)
	Stefano Delle Monache (s.dellemonache@tudelft.nl)
	Delphine Devallez
	Federico Fontana (federico.fontana@uniud.it)
	Laura Ottaviani
	Stefano Papetti (stefano.papetti@zhdk.ch)
	Pietro Polotti (pietro.polotti@conts.it)
	Matthias Rath
	Davide Rocchesso (davide.rocchesso@unipa.it)
	Stefania Serafin (sts@create.aau.dk)
	Marco Tiraboschi (marco.tiraboschi@unimi.it)
```

## Pure Data host signal reproduction

The noise and low-pass signal generation in `Reproduce.cxx` follows `src/d_osc.c` and `src/d_filter.c` from Pure Data 0.55-2.
Copyright (c) 1997-1999 Miller Puckette.

```text
This software is copyrighted by Miller Puckette and others.  The following
terms (the "Standard Improved BSD License") apply to all files associated with
the software unless explicitly disclaimed in individual files:

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.
3. The name of the author may not be used to endorse or promote
   products derived from this software without specific prior
   written permission.

THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR
BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF
THE POSSIBILITY OF SUCH DAMAGE.
```
