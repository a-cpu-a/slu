module;

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slu_NoConcepts

#include <utility>
#include <functional>
#include <format>

#include <slu/Ansi.hpp>
export module slu._test.test;

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
	slu::parse::VecInput in2{ slu::parse::sluCommon };
	slu::parse::BasicMpDbData mpDb;
	in2.genData.mpDb = { &mpDb };
	in2.genData.totalMp = {"hello_world"};
	auto f2 = slu::parse::parseFile(in2);

	slu::parse::Output out2(slu::parse::sluCommon);
	out2.db = std::move(in2.genData.mpDb);
	slu::parse::genFile(out2, {});

	slu::paint::SemOutput semOut2(in2);

	slu::paint::paintFile(semOut2, f2);
	slu::paint::toHtml(semOut2, true);

	slu::comp::compile({});

	auto vi2 = slu::visit::EmptyVisitor{ slu::parse::sluCommon };
	slu::visit::visitFile(vi2, f2);
}
