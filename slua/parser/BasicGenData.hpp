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
	using lang::LocalObjId;
	using lang::ModPathId;

	/*
		Names starting with $ are anonymous, and are followed by 8 raw bytes (representing anon name id)
		(All of those are not to be used by other modules, and are local only)

		When converting anonymous names into text, use this form: $(hex form of bytes, no leading zeros except for just 0)

		If a host asks for one of these names as text, then use one of these forms:
			Same as for normal text or
			__SLUAANON__(hex form of bytes, same as for normal text)

		(Hex forms in lowercase)
	*/
	constexpr std::string getAnonName(const size_t anonId)
	{
		_ASSERT(anonId != SIZE_MAX);
		std::string name(1 + sizeof(size_t), '$');
		name[1] = (anonId & 0xFF00000000000000) >> 56;
		name[2] = (anonId & 0xFF000000000000) >> 48;
		name[3] = (anonId & 0xFF0000000000) >> 40;
		name[4] = (anonId & 0xFF00000000) >> 32;
		name[5] = (anonId & 0xFF000000) >> 24;
		name[6] = (anonId & 0xFF0000) >> 16;
		name[7] = (anonId & 0xFF00) >> 8;
		name[8] = (anonId & 0xFF) >> 0;
		return name;
	}

	struct BasicModPathData
	{
		ModPath path;
		std::unordered_map<std::string, LocalObjId> name2Id;
		std::unordered_map<size_t, std::string> id2Name;

		constexpr LocalObjId get(const std::string& name)
		{
			if (!name2Id.contains(name))
			{
				const size_t res = name2Id.size();

				name2Id[name] = { res };
				id2Name[res] = name;

				return { res };
			}
			return name2Id[name];
		}
	};
	struct LuaMpDb
	{
		std::unordered_map<std::string, LocalObjId> name2Id;
		std::unordered_map<size_t, std::string> id2Name;

		LocalObjId get(const std::string& v)
		{
			if (!name2Id.contains(v))
			{
				const size_t res = name2Id.size();

				name2Id[v] = { res };
				id2Name[res] = v;

				return { res };
			}
			return name2Id[v];
		}

		std::string_view asSv(const MpItmIdV<false> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return id2Name.at(v.id.val);
		}
	};
	struct BasicMpDb
	{
		std::unordered_map<ModPath, ModPathId, lang::HashModPathView, lang::EqualModPathView> mp2Id;
		std::vector<BasicModPathData> mps = { {} };//Add 0, the unknown one

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

		std::string_view asSv(const MpItmIdV<true> v) const {
			if (v.id.val == SIZE_MAX)
				return {};//empty
			return mps[v.mp.id].id2Name.at(v.id.val);
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
		
		Sel<isSlua, LuaMpDb, BasicMpDb> mpDb;
		std::vector<BasicGenScopeV<isSlua>> scopes;
		std::vector<size_t> anonScopeCounts = {0};
		ModPath totalMp;

		constexpr BasicGenDataV()
		{
			scopes.emplace_back(SIZE_MAX);//anon id
		}

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

		std::string_view asSv(const MpItmIdV<isSlua> id) const {
			return mpDb.asSv(id);
		}

		constexpr void pushUnsafe() {
			scopes.back().safetyList.emplace_back(false, true);
		}
		constexpr void popSafety()
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
		constexpr void setUnsafe()
		{
			GenSafety& gs = scopes.back().safetyList.back();
			if(gs.forPop || gs.isSafe)
				scopes.back().safetyList.emplace_back(false);
		}
		constexpr void setSafe()
		{
			GenSafety& gs = scopes.back().safetyList.back();
			if (gs.forPop || !gs.isSafe)
				scopes.back().safetyList.emplace_back(true);
		}

		//For impl, lambda, scope, doExpr, things named '_'
		constexpr void pushAnonScope(){
			const size_t id = anonScopeCounts.back()++;
			const std::string name = getAnonName(id);
			addLocalObj(name);

			totalMp.push_back(name);
			scopes.push_back({id});
			anonScopeCounts.push_back(0);
		}
		//For func, macro, inline_mod, type?, ???
		constexpr void pushScope(const std::string& name) {
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
		constexpr void pushStat(StatementV<isSlua>&& stat){
			scopes.back().res.emplace_back(stat);
		}
		constexpr void addLocalObj(const std::string& name){
			scopes.back().objs.push_back(name);
		}

		constexpr std::optional<size_t> resolveLocalOpt(const std::string& name)
		{
			size_t scopeRevId = 0;
			for (const BasicGenScopeV<isSlua>& scope : std::views::reverse(scopes))
			{
				scopeRevId++;

				for (const std::string& var : scope.objs)
				{
					if (var == name)
						return scopeRevId;
				}
			}
			return {};
		}
		constexpr MpItmIdV<isSlua> resolveName(const std::string& name)
		{// Check if its local
			//either known local being indexed ORR unknown(potentially from a `use ::*`)
			const std::optional<size_t> v = resolveLocalOpt(name);
			if (v.has_value())
			{
				ModPathId mp = mpDb.get(
					ModPathView(totalMp).subspan(0, totalMp.size() - *v)
				);
				LocalObjId id = mpDb.mps[mp.id].get(name);
				return MpItmIdV<true>{id, mp};
			}
			return resolveUnknown(name);
		}
		constexpr MpItmIdV<isSlua> resolveName(const ModPath& name)
		{
			if (name.size() == 1)
				return resolveName(name[0]);//handles self implicitly!!!

			//either known local being indexed, super, crate, self ORR unknown(potentially from a `use ::*`)

			std::optional<size_t> v;
			if (name[0] == "self")
				v = scopes.size() - 1;//Pop all new ones
			else if (name[0] == "super")
				v = scopes.size();//Pop all new ones + self
			else if (name[0] == "crate")
				v = totalMp.size() - 1;//All but last
			else
				v = resolveLocalOpt(name[0]);

			if (v.has_value())
			{
				ModPath mpSum;
				mpSum.reserve((totalMp.size() - *v) + (name.size() - 1));

				for (size_t i = 0; i < totalMp.size() - *v; i++)
					mpSum.push_back(totalMp[i]);
				for (size_t i = 0; i < name.size() - 1; i++)
					mpSum.push_back(name[i]);

				ModPathId mp = mpDb.get(ModPathView(mpSum));

				LocalObjId id = mpDb.mps[mp.id].get(name.back());
				return MpItmIdV<true>{id,mp};
			}
			return resolveUnknown(name);
		}
		// .XXX, XXX, :XXX
		constexpr MpItmIdV<isSlua> resolveUnknown(const std::string& name)
		{
			if constexpr(isSlua)
			{
				LocalObjId id = mpDb.mps[0].get(name);
				return MpItmIdV<true>{id, { 0 }};
			}
			else
			{
				LocalObjId id = mpDb.get(name);
				return MpItmIdV<false>{id};
			}
		}
		constexpr MpItmIdV<isSlua> resolveUnknown(const ModPath& name)
		{
			ModPathId mp = mpDb.get(lang::UnknownModPathView{
				ModPathView(name).subspan(0, name.size() - 1) // All but last elem
			});
			LocalObjId id = mpDb.mps[mp.id].get(name.back());
			return MpItmIdV<true>{id,mp};
		}
		constexpr MpItmIdV<isSlua> resolveEmpty()
		{
			return MpItmIdV<isSlua>{SIZE_MAX};
		}
	};
}