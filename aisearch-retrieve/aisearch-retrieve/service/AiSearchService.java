package com.linkpoket.source.domain.aisearch.service;

import java.util.List;
import org.springframework.boot.autoconfigure.condition.ConditionalOnProperty;
import org.springframework.context.ApplicationEventPublisher;
import org.springframework.stereotype.Service;
import com.linkpoket.source.auth.oauth2.principal.PrincipalDetails;
import com.linkpoket.source.domain.aisearch.dto.request.AiSearchRequest;
import com.linkpoket.source.domain.aisearch.dto.response.AiSearchResponse;
import com.linkpoket.source.domain.aisearch.dto.response.AiSearchUsageDto;
import com.linkpoket.source.domain.aisearch.error.AiSearchErrorCode;
import com.linkpoket.source.domain.aisearch.event.AiSearchUsageEvent;
import com.linkpoket.source.domain.memberpageconnection.constant.PermissionType;
import com.linkpoket.source.domain.memberpageconnection.repository.MemberPageConnectionDataAccess;
import com.linkpoket.source.global.exception.BusinessException;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * 실시간 AI 지식 검색 질의응답 파이프라인의 통합 비즈니스 제어를 처리하는 구체 서비스 클래스입니다.
 */
@Slf4j
@Service
@RequiredArgsConstructor
@ConditionalOnProperty(name = "ai-search.enabled", havingValue = "true")
public class AiSearchService {

	private final MemberPageConnectionDataAccess pageConnectionDataAccess;
	private final AiSearchQuotaChecker quotaChecker;
	private final AiSearchRouter aiSearchRouter;
	private final ApplicationEventPublisher eventPublisher;

	/**
	 * 대화 맥락 없이 단발성 AI 검색을 수행합니다.
	 *
	 * @param request          사용자 질문문 및 검색 대상 페이지 식별자
	 * @param principalDetails 로그인 세션 정보 객체
	 * @return 종합 답변과 출처 리스트 DTO
	 */
	public AiSearchResponse search(final AiSearchRequest request, final PrincipalDetails principalDetails) {
		return search(request, principalDetails, List.of());
	}

	/**
	 * 사용자 요청 권한 및 일일 토큰/횟수 쿼터를 우선 검증하고 실시간 다단계 검색 라우팅을 수행하여 결과를 획득하고 이벤트를 비동기 전파합니다.
	 * 대화 이력이 주어지면 각 단계 LLM 호출에 함께 전달되어 꼬리질문이 앞선 맥락을 근거로 해석됩니다.
	 *
	 * @param request          사용자 질문문 및 검색 대상 페이지 식별자
	 * @param principalDetails 로그인 세션 정보 객체
	 * @param history          같은 대화 세션의 이전 질문/답변 이력 (단발 검색이면 빈 목록)
	 * @return 종합 답변과 출처 리스트 DTO
	 */
	public AiSearchResponse search(
		final AiSearchRequest request,
		final PrincipalDetails principalDetails,
		final List<ConversationTurn> history
	) {
		final Long memberId = principalDetails.getId();
		final Long pageId = request.pageId();

		// 1단계: 접근 제어 권한 및 일일 할당량 검증
		final List<Long> accessiblePageIds;
		if (pageId != null) {
			if (!pageConnectionDataAccess.hasPermissionAtLeast(memberId, pageId, PermissionType.VIEWER)) {
				throw new BusinessException(AiSearchErrorCode.UNAUTHORIZED_PAGE_ACCESS);
			}
			accessiblePageIds = List.of(pageId);
		} else {
			accessiblePageIds = pageConnectionDataAccess.findAllPageIdsByMemberId(memberId);
			if (accessiblePageIds.isEmpty()) {
				throw new BusinessException(AiSearchErrorCode.UNAUTHORIZED_PAGE_ACCESS);
			}
		}
		quotaChecker.validateDailyQuota(memberId);

		// 2~4단계: 라우터 호출
		final AiSearchResult result = aiSearchRouter.route(request.query(), accessiblePageIds, memberId, history);

		// 비동기 로깅 이벤트 발행
		eventPublisher.publishEvent(new AiSearchUsageEvent(
			memberId, request.query(), result.tokens(), result.latencyMs(), result.isSuccess()
		));

		return new AiSearchResponse(
			result.answer(),
			result.sources(),
			AiSearchUsageDto.builder()
				.totalCalls(1)
				.totalTokens(result.tokens())
				.build()
		);
	}
}