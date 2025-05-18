

//This file is used to test compilation

//Test macro, dont use, doesnt improve performace. (actually hurts it lol)
//#define Slua_NoConcepts
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

	slua::parse::VecInput in;
	const auto f = slua::parse::parseFile(in);

	slua::parse::Output out;
	out.db = std::move(in.genData.mpDb);
	slua::parse::genFile(out, {});

	slua::paint::SemOutput semOut(in);

	slua::paint::paintFile(semOut, f);
	slua::paint::toHtml(semOut, true);




	slua::parse::VecInput in2{ slua::parse::sluaCommon };
	const auto f2 =slua::parse::parseFile(in2);

	slua::parse::Output out2(slua::parse::sluaCommon);
	out2.db = std::move(in2.genData.mpDb);
	slua::parse::genFile(out2, {});

	slua::paint::SemOutput semOut2(in2);

	slua::paint::paintFile(semOut2, f2);
	slua::paint::toHtml(semOut2, true);
}