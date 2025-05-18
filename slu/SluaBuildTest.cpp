

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slu_NoConcepts
#include <slu/Include.hpp>
#include <slu/ErrorType.hpp>
#include <slu/Context.hpp>
#include <slu/MetaTableUtils.hpp>
#include <slu/WrapFunc.hpp>
#include <slu/Utils.hpp>

#include <slu/types/Converter.hpp>
#include <slu/types/ReadWrite.hpp>
#include <slu/types/TypeUtils.hpp>
#include <slu/types/UserData.hpp>
#include <slu/types/complex/AnyRef.hpp>
#include <slu/types/complex/OptFunc.hpp>
#include <slu/types/complex/Vector.hpp>
#include <slu/types/complex/Function.hpp>
#include <slu/types/basic/StackItem.hpp>
#include <slu/types/basic/Optional.hpp>
#include <slu/types/basic/RegistryRef.hpp>

#include <slu/parser/Parse.hpp>
#include <slu/parser/VecInput.hpp>
#include <slu/paint/Paint.hpp>
#include <slu/paint/PaintToHtml.hpp>
#include <slu/gen/Gen.hpp>
#include <slu/MetaTableUtils.hpp>


void _test()
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
	const auto f2 =slu::parse::parseFile(in2);

	slu::parse::Output out2(slu::parse::sluCommon);
	out2.db = std::move(in2.genData.mpDb);
	slu::parse::genFile(out2, {});

	slu::paint::SemOutput semOut2(in2);

	slu::paint::paintFile(semOut2, f2);
	slu::paint::toHtml(semOut2, true);
}