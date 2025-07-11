﻿/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form

#include <slu/Settings.hpp>
#include <slu/parser/Input.hpp>

namespace slu::parse
{
	//Here, so streamed outputs can be made
	template<class T>
	concept AnyOutput =
#ifdef Slu_NoConcepts
		true
#else
		AnyCfgable<T> && requires(T t) {

		//{ t.db } -> std::same_as<LuaMpDb>;
		
		//{ t.resolveLocal(parse::LocalId()) } -> std::same_as<parse::MpItmIdV<>&>;
		//{ t.pushL(parse::LocalId()) } -> std::same_as<parse::MpItmIdV<>>;
		{ t.popLocals() } -> std::same_as<void>;

		{ t.add(char(1)) } -> std::same_as<T&>;
		{ t.add("aa") } -> std::same_as<T&>;
		{ t.add(std::string_view()) } -> std::same_as<T&>;
		{ t.add(std::span<const uint8_t>()) } -> std::same_as<T&>;

		{ t.template addNewl<true>(char(1)) } -> std::same_as<T&>;
		{ t.addNewl(char(1)) } -> std::same_as<T&>;
		{ t.addNewl(std::string_view()) } -> std::same_as<T&>;
		{ t.addNewl(std::span<const uint8_t>()) } -> std::same_as<T&>;

		{ t.addMultiline(std::span<const uint8_t>(),size_t()) } -> std::same_as<T&>;
		{ t.addMultiline(std::span<const uint8_t>()) } -> std::same_as<T&>;

		{ t.newLine() } -> std::same_as<T&>;
		{ t.template newLine<true>() } -> std::same_as<T&>;

		{ t.addIndent() } -> std::same_as<T&>;

		{ t.tabUp() } -> std::same_as<T&>;
		{ t.unTab() } -> std::same_as<T&>;

		{ t.template tabUpNewl<true>() } -> std::same_as<T&>;
		{ t.tabUpNewl() } -> std::same_as<T&>;
		{ t.unTabNewl() } -> std::same_as<T&>;

		{ t.tabUpTemp() } -> std::same_as<T&>;
		{ t.unTabTemp() } -> std::same_as<T&>;

		{ t.wasSemicolon } -> std::same_as<bool&>;
	}
#endif // Slu_NoConcepts
	;

	inline void updateLinePos(size_t& curLinePos, std::span<const uint8_t> sp)
	{
		for (const uint8_t ch : sp)
		{
			if (ch == '\r' || ch == '\n')
				curLinePos = 0;
			else
				curLinePos++;
		}
	}

	template<AnySettings SettingsT = Setting<void>>
	struct Output
	{
		constexpr Output(SettingsT) {}
		constexpr Output() = default;
		constexpr static bool SLU_SYN = SettingsT() & sluSyn;

		constexpr static SettingsT settings()
		{
			return SettingsT();
		}


		Sel<SLU_SYN, LuaMpDb, BasicMpDb> db;
		std::vector<const parse::LocalsV<SLU_SYN>*> localStack;

		std::vector<uint8_t> text;
		uint64_t tabs=0;
		size_t curLinePos = 0;
		bool tempDownTab = false;
		bool wasSemicolon = false;


		auto resolveLocal(parse::LocalId local) {
			return localStack.back()->at(local.v);
		}
		void pushLocals(const parse::LocalsV<SLU_SYN>& locals) {
			localStack.push_back(&locals);
		}
		void popLocals() {
			localStack.pop_back();
		}

		Output& add(const char ch) 
		{
			text.push_back(ch);
			curLinePos++;
			wasSemicolon = false;
			return *this;
		}
		Output& add(const std::string_view sv) 
		{
			text.insert(text.end(), sv.begin(), sv.end());
			curLinePos+= sv.size();
			wasSemicolon = false;
			return *this;
		}
		template<size_t N>
		Output& add(const char(&sa)[N]) {
			text.insert(text.end(), sa, &(sa[N - 1]));//skip null
			curLinePos += N-1;//skip null
			wasSemicolon = false;
			return *this;
		}
		Output& add(std::span<const uint8_t> sp) 
		{
			text.insert(text.end(), sp.begin(), sp.end());
			curLinePos += sp.size();
			wasSemicolon = false;
			return *this;
		}
		Output& addMultiline(std::span<const uint8_t> sp,const size_t linePos)
		{
			text.insert(text.end(), sp.begin(), sp.end());
			curLinePos = linePos;
			wasSemicolon = false;
			return *this;
		}
		Output& addMultiline(std::span<const uint8_t> sp)
		{
			text.insert(text.end(), sp.begin(), sp.end());
			updateLinePos(curLinePos,sp);
			wasSemicolon = false;
			return *this;
		}

		Output& addIndent()
		{
			text.insert(text.end(), tabs, '\t');
			curLinePos += tabs*4;
			return *this;
		}
		Output& tabUp()
		{
			tabs++;
			return *this;
		}
		Output& unTab() 
		{
			tabs--;
			return *this;
		}
		Output& tabUpTemp()
		{
			tempDownTab = false;
			return *this;
		}
		Output& unTabTemp()
		{
			tempDownTab = true;
			return *this;
		}

		template<bool runIndent=true>
		Output& newLine()
		{
			text.push_back('\n');

			if constexpr (runIndent)
				addIndent();

			curLinePos = 0;
			return *this;
		}

		template<bool runIndent = true>
		Output& tabUpNewl() {
			return tabUp().template newLine<runIndent>();
		}
		Output& unTabNewl() {
			return unTab().newLine();
		}

		template<bool runIndent = true>
		Output& addNewl(const char ch) {
			return add(ch).template newLine<runIndent>();
		}
		Output& addNewl(const std::string_view sv) {
			return add(sv).newLine();
		}
		template<size_t N>
		Output& addNewl(const char (&sa)[N]) {
			return add(sa).newLine();
		}
		Output& addNewl(std::span<const uint8_t> sp) {
			return add(sp).newLine();
		}
	};
}