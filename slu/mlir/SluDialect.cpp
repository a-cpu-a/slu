/*
    Slu language compiler, a computer program compiler.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
#include "slu/mlir/SluDialect.h"

using namespace mlir;
using namespace slu_dial;

#include "slu/mlir/SluOpsDialect.cpp.inc"


void slu_dial::SluDialect::initialize()
{
	addOperations<
#define GET_OP_LIST
#include "slu/mlir/SluOps.cpp.inc"
	    >();
}