/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <ranges>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

namespace slua::parse
{
	struct ModPathId
	{
		size_t val;
	};
	struct MpItmId
	{
		ModPathId mp;
		size_t valId;
	};


	struct GenSafety
	{
		bool isSafe : 1 = false;
		bool forPop : 1 = false;
	};
	template<bool isSlua>
	struct BasicGenScopeV
	{
		std::string name;//empty -> anon

		std::vector<std::string> objs;

		std::vector<GenSafety> safetyList;

		BlockV<isSlua> res;
	};
	template<bool isSlua>
	struct BasicGenDataV
	{
		//ParsedFileV<isSlua> out; //TODO_FOR_COMPLEX_GEN_DATA: field is ComplexOutData&, and needs to be obtained from shared mutex
		//TODO: basic name DB, to allow similar code for complex & basic

		std::vector<BasicGenScopeV<isSlua>> scopes;
		ModPath root;

		/*
		All local names (no ::'s) are defined in THIS file, or from a `use ::*` (potentialy std::prelude::*)
		This means any 'use' things that reference ??? stuff, can only be other global crates OR a sub-path of a `use ::*`
		
		Since we dont know if 'mylib' is a crate, or a module inside something that was `use ::*`-ed,
		we have to wait until all ::* modules have been parsed.
		OR, we have to let another pass figure it out. (The simpler option)

		Recursive ::* uses, might force the multi-pass aproach?
		Or would be impossible.
		Which is why ::* should enforce order and no recursive use.

		How to handle 2 `use ::*`'s?

		If there are 2 star uses, is it required to union both of them, into 1 symbol id?
		Are we forced to make a new symbol after every combination of star uses? No (VVV)
		Is it better to just use unknown_modpath?
		^ Yes, doing so, will simplify the ability to add or remove the default used files

		*/

		void pushUnsafe() {
			scopes.back().safetyList.emplace_back(false, true);
		}
		void popSafety() 
		{
			std::vector<GenSafety>& safetyList = scopes.back().safetyList;
			size_t popCount = 1;

			for (const GenSafety gs : std::views::reverse(safetyList))
			{
				if (gs.forPop)
					break;
				popCount++;
			}
			safetyList.erase(safetyList.end()-popCount, safetyList.end());
		}

		void setUnsafe() 
		{
			GenSafety& gs = scopes.back().safetyList.back();
			if(gs.forPop || gs.isSafe)
				scopes.back().safetyList.emplace_back(false);
		}
		void setSafe()
		{
			GenSafety& gs = scopes.back().safetyList.back();
			if (gs.forPop || !gs.isSafe)
				scopes.back().safetyList.emplace_back(true);
		}

		//For impl, lambda, scope, doExpr
		void pushAnonScope(){
			scopes.push_back({""});
		}
		//For func, macro, inline_mod, type?, ???
		void pushScope(const std::string& name) {
			scopes.push_back({ name });
		}
		BlockV<isSlua> popScope() {
			BlockV<isSlua> res = std::move(scopes.back().res);
			scopes.pop_back();
			return res;
		}
		void pushStat(StatementV<isSlua>&& stat){
			scopes.back().res.emplace_back(stat);
		}
		void addLocalObj(const std::string& name){
			scopes.back().objs.push_back(name);
		}

		MpItmId resolveName(const std::string& name)const {}
		MpItmId resolveName(const ModPath& name)const {}
	};
}