/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>
#include <optional>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include "Enums.h"

namespace sluaParse
{
	struct Position
	{
		size_t line;
		size_t index;
	};

	struct Scope
	{
		std::vector<struct Variable> variables;
		std::vector<struct Function> functions;

		Scope* owner;//nullptr -> this is the global scope

		Position start;
		Position end;
	};

	struct Variable
	{
		std::string name;
		Scope* owner;//nullptr -> error

		Position place;
		//size_t typeId;
	};

	struct Parameter
	{
		std::string name;
		//size_t typeId;
	};

	struct Statement
	{
		Position place;


		StatementType type = StatementType::SEMICOLON;
	};

	struct Expression
	{
		Position place;

		//todo: many things possible

		ExpType type = ExpType::NIL;
	};

	using ExpList = std::vector<Expression>;

	struct Block
	{
		std::vector<Statement> statList;
		std::optional<ExpList> ret;

		Scope scope;

		Position start;
		Position end;
	};

	struct Function : Variable
	{
		std::vector<Parameter> params;
		Block block;
	};
}