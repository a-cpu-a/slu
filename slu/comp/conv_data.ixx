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
module;

//#include <slu/comp/mlir/Pch.hpp>
export module slu.comp.conv_data;

import slu.ast.mp_data;
import slu.ast.state;
import slu.comp.cfg;

namespace slu::comp
{
	export struct CommonConvData
	{
		const CompCfg& cfg;
		const parse::BasicMpDbData& sharedDb;
		const parse::Stat* stat;
		std::string_view filePath;
	};
} //namespace slu::comp