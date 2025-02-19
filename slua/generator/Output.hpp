/*
** See Copyright Notice inside Include.hpp
*/
#pragma once

#include <string>
#include <span>
#include <vector>

//https://www.lua.org/manual/5.4/manual.html
//https://en.wikipedia.org/wiki/Extended_Backus%E2%80%93Naur_form
//https://www.sciencedirect.com/topics/computer-science/backus-naur-form


namespace sluaParse
{
	//Here, so streamed outputs can be made
	template<class T>
	concept AnyOutput = requires(T t) {
		{ t.add(char(1)) } -> std::same_as<T&>;
		{ t.add(std::string_view()) } -> std::same_as<T&>;
		{ t.add(std::span<const uint8_t>()) } -> std::same_as<T&>;
		{ t.addNewl(char(1)) } -> std::same_as<T&>;
		{ t.addNewl(std::string_view()) } -> std::same_as<T&>;
		{ t.addNewl(std::span<const uint8_t>()) } -> std::same_as<T&>;
		{ t.addMultiline(std::span<const uint8_t>(),size_t()) } -> std::same_as<T&>;
		{ t.addMultiline(std::span<const uint8_t>()) } -> std::same_as<T&>;

		{ t.newLine() } -> std::same_as<T&>;

		{ t.addIndent() } -> std::same_as<T&>;

		{ t.tabUp() } -> std::same_as<T&>;
		{ t.unTab() } -> std::same_as<T&>;

		{ t.tabUpNewl() } -> std::same_as<T&>;
		{ t.unTabNewl() } -> std::same_as<T&>;

		{ t.tabUpTemp() } -> std::same_as<T&>;
		{ t.unTabTemp() } -> std::same_as<T&>;
	};

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

	struct Output
	{
		std::vector<uint8_t> text;
		uint64_t tabs=0;
		size_t curLinePos = 0;
		bool tempDownTab = false;

		Output& add(const char ch) 
		{
			text.push_back(ch);
			curLinePos++;
			return *this;
		}
		Output& add(const std::string_view sv) 
		{
			text.insert(text.end(), sv.begin(), sv.end());
			curLinePos+= sv.size();
			return *this;
		}
		Output& add(std::span<const uint8_t> sp) 
		{
			text.insert(text.end(), sp.begin(), sp.end());
			curLinePos += sp.size();
			return *this;
		}
		Output& addMultiline(std::span<const uint8_t> sp,const size_t linePos)
		{
			text.insert(text.end(), sp.begin(), sp.end());
			curLinePos = linePos;
			return *this;
		}
		Output& addMultiline(std::span<const uint8_t> sp)
		{
			text.insert(text.end(), sp.begin(), sp.end());
			updateLinePos(curLinePos,sp);
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

		Output& newLine()
		{
			text.push_back('\n');
			addIndent();
			curLinePos = 0;
			return *this;
		}

		Output& tabUpNewl() {
			return tabUp().newLine();
		}
		Output& unTabNewl() {
			return unTab().newLine();
		}

		Output& addNewl(const char ch) {
			return add(ch).newLine();
		}
		Output& addNewl(const std::string_view sv) {
			return add(sv).newLine();
		}
		Output& addNewl(std::span<const uint8_t> sp) {
			return add(sp).newLine();
		}
	};
}