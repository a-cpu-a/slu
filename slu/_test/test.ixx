/*
    A program file.
    Copyright (C) 2026 a-cpu-a <any1word@proton.me>

    This file is part of Slu-c.

    Slu-c is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Slu-c is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with Slu-c.  If not, see <https://www.gnu.org/licenses/>.

      SPDX-License-Identifier: AGPL3.0-or-later
*/
module;

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slu_NoConcepts

#include <format>
#include <functional>
#include <utility>

#include <slu/Ansi.hpp>
export module zzz_internal_slu.test.test;

import slu.settings;
import slu.ast.mp_data;
import slu.comp.compile;
import slu.gen.gen;
import slu.gen.output;
import slu.mlvl.type_inf_check;
import slu.paint.paint;
import slu.paint.sem_output;
import slu.paint.to_html;
import slu.parse.parse;
import slu.parse.vec_input;
import slu.visit.empty;
import slu.visit.visit;


void ____test()
{
	slu::parse::VecInput in2{slu::parse::sluCommon};
	slu::parse::BasicMpDbData mpDb;
	in2.genData.mpDb = {&mpDb};
	in2.genData.totalMp = {"hello_world"};
	auto f2 = slu::parse::parseFile(in2);

	slu::parse::Output out2(slu::parse::sluCommon);
	out2.db = std::move(in2.genData.mpDb);
	slu::parse::genFile(out2, {});

	slu::paint::SemOutput semOut2(in2);

	slu::paint::paintFile(semOut2, f2);
	slu::paint::toHtml(semOut2, true);

	slu::comp::compile({});

	auto vi2 = slu::visit::EmptyVisitor{slu::parse::sluCommon};
	slu::visit::visitFile(vi2, f2);
}
