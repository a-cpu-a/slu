/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <cstdint>
#include <ranges>
#include <unordered_map>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slua/parser/State.hpp>
#include <slua/parser/Input.hpp>

namespace slua::parse
{
	/*
		Names starting with $ are anonymous, and are followed by 8 raw bytes (representing anon name id)
		(All of those are not to be used by other modules, and are local only)

		When converting anonymous names into text, use this form: $(hex form of bytes, no leading zeros except for just 0)

		If a host asks for one of these names as text, then use one of these forms:
			Same as for normal text or
			__SLUAANON__(hex form of bytes, same as for normal text)

		(Hex forms in lowercase)
	*/
	std::string getAnonName(const size_t anonId)
	{
		_ASSERT(anonId != SIZE_MAX);
		std::string name(1 + sizeof(size_t), '$');
		name[1] = (anonId & 0xFF000000) >> 24;
		name[2] = (anonId & 0xFF0000) >> 16;
		name[3] = (anonId & 0xFF00) >> 8;
		name[4] = (anonId & 0xFF) >> 0;
		return name;
	}

	struct BasicModPathData
	{
		ModPath path;
		std::unordered_map<std::string, LocalObjId> name2Id;

		LocalObjId get(const std::string& name)
		{
			if (!name2Id.contains(name))
			{
				const size_t res = name2Id.size();

				name2Id[name] = { res };

				return { res };
			}
			return name2Id[name];
		}
	};
	struct BasicMpDb
	{
		std::unordered_map<ModPath, ModPathId, lang::HashModPathView, lang::EqualModPathView> mp2Id;
		std::vector<BasicModPathData> mps;

		ModPathId get(lang::AnyModPathView auto path)
		{
			if (!mp2Id.contains(path))
			{
				const size_t res = mps.size();

				mp2Id.find(path)->second = { res };
				mps.emplace_back(path);

				return { res };
			}
			return mp2Id.find(path)->second;
		}
	};

	struct GenSafety
	{
		bool isSafe : 1 = false;
		bool forPop : 1 = false;
	};
	template<bool isSlua>
	struct BasicGenScopeV
	{
		size_t anonId;//size_max -> not anon

		std::vector<std::string> objs;

		std::vector<GenSafety> safetyList;

		BlockV<isSlua> res;

	};
	template<bool isSlua>
	struct BasicGenDataV
	{
		//ParsedFileV<isSlua> out; //TODO_FOR_COMPLEX_GEN_DATA: field is ComplexOutData&, and needs to be obtained from shared mutex
		//TODO: basic name DB, to allow similar code for complex & basic

		BasicMpDb mpDb;
		std::vector<BasicGenScopeV<isSlua>> scopes = { {SIZE_MAX} };
		std::vector<size_t> anonScopeCounts = {0};
		ModPath totalMp;

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
			const size_t id = anonScopeCounts.back()++;
			const std::string name = getAnonName(id);
			addLocalObj(name);

			totalMp.push_back(name);
			scopes.push_back({id});
			anonScopeCounts.push_back(0);
		}
		//For func, macro, inline_mod, type?, ???
		void pushScope(const std::string& name) {
			addLocalObj(name);

			totalMp.push_back(name);
			scopes.push_back({ SIZE_MAX });
			anonScopeCounts.push_back(0);
		}
		BlockV<isSlua> popScope() {
			BlockV<isSlua> res = std::move(scopes.back().res);
			scopes.pop_back();
			totalMp.pop_back();
			anonScopeCounts.pop_back();
			return res;
		}
		void pushStat(StatementV<isSlua>&& stat){
			scopes.back().res.emplace_back(stat);
		}
		void addLocalObj(const std::string& name){
			scopes.back().objs.push_back(name);
		}

		MpItmId<isSlua> resolveName(const std::string& name)
		{// Check if its local
			//either known local being indexed ORR unknown(potentially from a `use ::*`)

			//TODO

			return resolveUnknownName(name);
		}
		MpItmId<isSlua> resolveName(const ModPath& name)
		{
			if (name.size() == 1)
				return resolveName(name[0]);
			//either known local being indexed, super, crate, self ORR unknown(potentially from a `use ::*`)

			//TODO
			return resolveUnknownName(name);
		}
		// .XXX, XXX, :XXX
		MpItmId<isSlua> resolveUnknownName(const std::string& name)
		{
			LocalObjId id = mpDb.mps[0].get(name);
			return MpItmId<true>{{0}, id};
		}
		MpItmId<isSlua> resolveUnknownName(const ModPath& name)
		{
			ModPathId mp = mpDb.get(lang::UnknownModPathView{
				ModPathView(name).subspan(0, name.size() - 1) 
			});//All but last elem

			LocalObjId id = mpDb.mps[0].get(name.back());
			return MpItmId<true>{mp, id};
		}
	};
}