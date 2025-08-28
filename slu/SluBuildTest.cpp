

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slu_NoConcepts

#include <variant>
#include <array>
#include <span>
#include <string>
#include <utility>
#include <string_view>
#include <unordered_map>

#include <slu/Ansi.hpp>
#include <slu/parse/Parse.hpp>
#include <slu/visit/Visit.hpp>
#include <slu/gen/Gen.hpp>
#include <slu/comp/Compile.hpp>
#include <slu/mlvl/TypeInfCheck.hpp>
import slu.paint.paint;
import slu.paint.to_html;
import slu.parse.vec_input;


void ____test()
{
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
