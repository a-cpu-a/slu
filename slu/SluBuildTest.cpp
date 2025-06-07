

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slu_NoConcepts

#include <slu/ext/lua/luaconf.h>
#include <slu/parser/Parse.hpp>
#include <slu/parser/VecInput.hpp>
#include <slu/paint/Paint.hpp>
#include <slu/paint/PaintToHtml.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/gen/Gen.hpp>
#include <slu/MetaTableUtils.hpp>
#include <slu/comp/Compile.hpp>


static void _test()
{

	slu::parse::VecInput in;
	const auto f = slu::parse::parseFile(in);

	slu::parse::Output out;
	out.db = std::move(in.genData.mpDb);
	slu::parse::genFile(out, {});

	slu::paint::SemOutput semOut(in);

	slu::paint::paintFile(semOut, f);
	slu::paint::toHtml(semOut, true);




	slu::parse::VecInput in2{ slu::parse::sluCommon };
	slu::parse::BasicMpDbData mpDb;
	in2.genData.mpDb = { &mpDb };
	in2.genData.totalMp = {"hello_world"};
	auto f2 =slu::parse::parseFile(in2);

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