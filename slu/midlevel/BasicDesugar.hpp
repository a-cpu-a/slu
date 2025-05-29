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

namespace slu::mlvl
{
	inline void basicDesugar(parse::ParsedFileV<true>& pf)
	{
		//TODO: Implement the conversion logic here
		//TODO: basic desugaring:
		//TODO: operators
		//TODO: auto-drop?
		//TODO: for/while/repeat loops
	}
}