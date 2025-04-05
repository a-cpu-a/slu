/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <unordered_set>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>
#include <slua/parser/adv/SkipSpace.hpp>
#include <slua/parser/adv/RequireToken.hpp>
#include <slua/parser/adv/ReadName.hpp>

namespace sluaParse
{
	template<AnyInput In>
	inline std::string readLabel(In& in)
	{
		//label ::= ‘::’ Name ‘::’
		//SL label ::= ‘:::’ Name ‘:’

		requireToken(in, sel<In>("::", ":::"));

		const std::string res = readName(in);

		requireToken(in, sel<In>("::", ":"));

		return res;
	}

	template<AnyInput In>
	inline bool readTypeStat(In& in, StatementData<In>& outData, const ExportData exported)
	{
		if (checkReadTextToken(in, "type"))
		{
			StatementType::TYPE res{};
			res.exported = exported;

			res.name = readName(in);
			requireToken(in, "=");
			res.ty = readType(in);

			outData = std::move(res);
			return true;
		}
		return false;
	}
}