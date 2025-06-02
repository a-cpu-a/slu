/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <vector>
#include <optional>
#include <thread>
#include <variant>
#include <slu/lang/BasicState.hpp>
#include <slu/parser/State.hpp>
#include <slu/visit/Visit.hpp>

namespace slu::mlvl
{
	struct MlvlVisitor : visit::EmptyVisitor<decltype(parse::sluCommon)>
	{
		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: operators
		//TODO: auto-drop?
		//TODO: for/while/repeat loops

	};

	inline void basicDesugar(parse::ParsedFileV<true>& itm)
	{
		MlvlVisitor vi{};
		visit::visitFile(vi, itm);
	}
}